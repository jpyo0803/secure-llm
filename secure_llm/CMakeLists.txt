cmake_minimum_required(VERSION 3.10)
project(secure_llm)

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_SGX "Enable SGX mode" OFF)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/common)

# Collect common source files
file(GLOB COMMON_SOURCES "${CMAKE_SOURCE_DIR}/common/*.cpp")

# Secure LLM source files
set(SECURE_LLM_SOURCES ${CMAKE_SOURCE_DIR}/secure_llm.cpp)

if (ENABLE_SGX)
    # SGX SDK Settings
    set(SGX_SDK /opt/intel/sgxsdk CACHE PATH "Path to SGX SDK")
    set(SGX_MODE HW CACHE STRING "SGX Mode (HW or SIM)")
    set(SGX_ARCH x64 CACHE STRING "SGX Architecture")
    set(SGX_DEBUG 0 CACHE STRING "Enable debugging")
    set(SGX_PRERELEASE 1 CACHE STRING "Enable pre-release")

    # Set x86 if 32bit system
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(SGX_ARCH x86)
    elseif(CMAKE_CXX_FLAGS MATCHES "-m32")
        set(SGX_ARCH x86)
    endif()

    if(SGX_ARCH STREQUAL "x86")
        set(SGX_COMMON_CFLAGS -m32)
        set(SGX_LIBRARY_PATH ${SGX_SDK}/lib)
        set(SGX_ENCLAVE_SIGNER ${SGX_SDK}/bin/x86/sgx_sign)
        set(SGX_EDGER8R ${SGX_SDK}/bin/x86/sgx_edger8r)
    else()
        set(SGX_COMMON_CFLAGS -m64)
        set(SGX_LIBRARY_PATH ${SGX_SDK}/lib64)
        set(SGX_ENCLAVE_SIGNER ${SGX_SDK}/bin/x64/sgx_sign)
        set(SGX_EDGER8R ${SGX_SDK}/bin/x64/sgx_edger8r)
    endif()

    if(SGX_DEBUG)
        if(SGX_PRERELEASE)
            message(FATAL_ERROR "Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!")
        endif()
        list(APPEND SGX_COMMON_CFLAGS -O3 -g)
    else()
        list(APPEND SGX_COMMON_CFLAGS -O3)
    endif()

    # Run in real hardware or simulation mode
    if(SGX_MODE STREQUAL "HW")
        set(Urts_Library_Name sgx_urts)
    else()
        set(Urts_Library_Name sgx_urts_sim)
    endif()

    set(App_Cpp_Files ${CMAKE_SOURCE_DIR}/App/enclave_bridge.cpp)
    set(App_Include_Paths -IApp -I${SGX_SDK}/include -I${TF_INC} -I${TF_INC}/external/nsync/public)

    set(App_C_Flags ${SGX_COMMON_CFLAGS} -fPIC -Wno-attributes ${App_Include_Paths} -fopenmp)

    if(SGX_DEBUG)
        list(APPEND App_C_Flags -DDEBUG -UNDEBUG -UEDEBUG)
    elseif(SGX_PRERELEASE)
        list(APPEND App_C_Flags -DNDEBUG -DEDEBUG -UDEBUG)
    else()
        list(APPEND App_C_Flags -DNDEBUG -UEDEBUG -UDEBUG)
    endif()

    set(App_Cpp_Flags ${App_C_Flags} -std=c++17 -shared -DUSE_SGX)
    set(App_Link_Flags ${SGX_COMMON_CFLAGS} -L${SGX_LIBRARY_PATH} -l${Urts_Library_Name} -L${TF_LIB} -pthread -fopenmp)

    if(SGX_MODE STREQUAL "HW")
        list(APPEND App_Link_Flags -lsgx_uae_service)
    else()
        list(APPEND App_Link_Flags -lsgx_uae_service_sim)
    endif()

    set(App_Cpp_Objects ${CMAKE_SOURCE_DIR}/App/enclave_bridge.so)
    set(App_Name secure_llm)

    # Enclave Settings
    if(SGX_MODE STREQUAL "HW")
        set(Trts_Library_Name sgx_trts)
        set(Service_Library_Name sgx_tservice)
    else()
        set(Trts_Library_Name sgx_trts_sim)
        set(Service_Library_Name sgx_tservice_sim)
    endif()
    set(Crypto_Library_Name sgx_tcrypto)

    set(My_Cpp_Files ${CMAKE_SOURCE_DIR}/common/aes_stream.cpp ${CMAKE_SOURCE_DIR}/common/tensor.cpp)
    set(Enclave_Cpp_Files ${CMAKE_SOURCE_DIR}/Enclave/Enclave.cpp ${CMAKE_SOURCE_DIR}/secure_llm.cpp ${My_Cpp_Files})
    set(Enclave_Include_Paths -I${CMAKE_SOURCE_DIR}/Enclave -I${SGX_SDK}/include -I${SGX_SDK}/include/tlibc -I${SGX_SDK}/include/libcxx -I${CMAKE_SOURCE_DIR}/common -I/usr/lib/gcc/x86_64-linux-gnu/11/include)

    if(CMAKE_C_COMPILER_VERSION VERSION_LESS "4.9")
        set(Enclave_C_Flags ${SGX_COMMON_CFLAGS} -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections -fstack-protector)
    else()
        set(Enclave_C_Flags ${SGX_COMMON_CFLAGS} -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections -fstack-protector-strong)
    endif()

    list(APPEND Enclave_C_Flags ${Enclave_Include_Paths})
    set(Enclave_Cpp_Flags ${Enclave_C_Flags} -std=c++17 -nostdinc++ -march=native -maes)

    set(Enclave_Link_Flags ${SGX_COMMON_CFLAGS} -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L${SGX_LIBRARY_PATH}
        -Wl,--whole-archive -l${Trts_Library_Name} -Wl,--no-whole-archive
        -Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l${Crypto_Library_Name} -l${Service_Library_Name} -Wl,--end-group
        -Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined
        -Wl,-pie,-eenclave_entry -Wl,--export-dynamic
        -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/Enclave/Enclave.lds)

    set(Enclave_Name enclave.so)
    set(Signed_Enclave_Name enclave.signed.so)
    set(Enclave_Config_File ${CMAKE_SOURCE_DIR}/Enclave/Enclave.config.xml)

    if(SGX_MODE STREQUAL "HW")
        if(SGX_DEBUG)
            set(Build_Mode HW_DEBUG)
        elseif(SGX_PRERELEASE)
            set(Build_Mode HW_PRERELEASE)
        else()
            set(Build_Mode HW_RELEASE)
        endif()
    else()
        if(SGX_DEBUG)
            set(Build_Mode SIM_DEBUG)
        elseif(SGX_PRERELEASE)
            set(Build_Mode SIM_PRERELEASE)
        else()
            set(Build_Mode SIM_RELEASE)
        endif()
    endif()

    # App Objects
    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/App/Enclave_u.c
        COMMAND ${SGX_EDGER8R} --untrusted ${CMAKE_SOURCE_DIR}/Enclave/Enclave.edl --search-path ${CMAKE_SOURCE_DIR}/Enclave --search-path ${SGX_SDK}/include
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/App
        COMMENT "GEN  =>  ${CMAKE_SOURCE_DIR}/App/Enclave_u.c"
    )

    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/App/Enclave_u.o
        COMMAND ${CMAKE_C_COMPILER} ${App_C_Flags} -c ${CMAKE_SOURCE_DIR}/App/Enclave_u.c -o ${CMAKE_SOURCE_DIR}/App/Enclave_u.o
        DEPENDS ${CMAKE_SOURCE_DIR}/App/Enclave_u.c
        COMMENT "CC   <=  ${CMAKE_SOURCE_DIR}/App/Enclave_u.c"
    )

    add_library(enclave_bridge SHARED ${CMAKE_SOURCE_DIR}/App/enclave_bridge.cpp ${CMAKE_SOURCE_DIR}/App/Enclave_u.o)
    target_compile_options(enclave_bridge PRIVATE ${App_Cpp_Flags})
    target_link_libraries(enclave_bridge PRIVATE ${App_Link_Flags})
    set_target_properties(enclave_bridge PROPERTIES OUTPUT_NAME "enclave_bridge")

    add_executable(${App_Name} ${CMAKE_SOURCE_DIR}/App/Enclave_u.o ${CMAKE_SOURCE_DIR}/App/enclave_bridge.so)
    target_link_libraries(${App_Name} PRIVATE enclave_bridge ${App_Link_Flags})
    set_target_properties(${App_Name} PROPERTIES OUTPUT_NAME "secure_llm")

    # Enclave Objects
    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.c
        COMMAND ${SGX_EDGER8R} --trusted ${CMAKE_SOURCE_DIR}/Enclave/Enclave.edl --search-path ${CMAKE_SOURCE_DIR}/Enclave --search-path ${SGX_SDK}/include
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/Enclave
        COMMENT "GEN  =>  ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.c"
    )

    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.o
        COMMAND ${CMAKE_C_COMPILER} ${Enclave_C_Flags} -c ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.c -o ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.o
        DEPENDS ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.c
        COMMENT "CC   <=  ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.c"
    )

    add_library(enclave OBJECT ${CMAKE_SOURCE_DIR}/Enclave/Enclave_t.o ${Enclave_Cpp_Files})
    target_compile_options(enclave PRIVATE ${Enclave_Cpp_Flags})

    add_library(enclave_shared SHARED $<TARGET_OBJECTS:enclave>)
    target_link_libraries(enclave_shared PRIVATE ${Enclave_Link_Flags})
    set_target_properties(enclave_shared PROPERTIES OUTPUT_NAME ${Enclave_Name})

    add_custom_command(OUTPUT ${Signed_Enclave_Name}
        COMMAND ${SGX_ENCLAVE_SIGNER} sign -key ${CMAKE_SOURCE_DIR}/Enclave/Enclave_private.pem -enclave ${Enclave_Name} -out ${Signed_Enclave_Name} -config ${Enclave_Config_File}
        DEPENDS ${Enclave_Name}
        COMMENT "SIGN =>  ${Signed_Enclave_Name}"
    )

else()
    # Non-SGX version
    include_directories(${CMAKE_SOURCE_DIR}/common)

    add_library(cipher_cpp SHARED ${COMMON_SOURCES})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

    target_compile_options(cipher_cpp PRIVATE -mavx512f -maes -march=native)

    add_executable(secure_llm ${SECURE_LLM_SOURCES})
    target_link_libraries(secure_llm PRIVATE cipher_cpp)
    target_compile_options(secure_llmp PRIVATE -mavx512f -maes -march=native)
endif()
