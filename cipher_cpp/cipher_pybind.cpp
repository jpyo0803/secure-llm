#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cipher.h"
#include <iostream>
#include <stdint.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
py::array_t<T> sum_by_row_wrapper(py::array_t<T, py::array::c_style | py::array::forcecast> input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("Input should be a 2D array");
    }

    int M = input.shape(0);
    int N = input.shape(1);

    auto result = py::array_t<T>(M);
   // auto r = result.mutable_unchecked<1>(); // For direct access to elements
    auto r = result.mutable_data();
    // Allocate T** for input
    T** in = new T*[M];
    for (int i = 0; i < M; ++i) {
      in[i] = static_cast<T*>(input.mutable_data()) + i * N;
    }

    // Call the original function
    jpyo0803::SumByRow(in, result.mutable_data(), M, N);

    // Clean up
    delete[] in;

    return result;
}

template <typename T>
py::array_t<T> sum_by_col_wrapper(py::array_t<T, py::array::c_style | py::array::forcecast> input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("Input should be a 2D array");
    }

    int M = input.shape(0);
    int N = input.shape(1);

    auto result = py::array_t<T>(N);
    std::fill_n(result.mutable_data(), N, 0); // Explicitly set to zero
    auto r = result.mutable_data();

    // Allocate int** for input
    T** in = new T*[M];
    for (int i = 0; i < M; ++i) {
        in[i] = &input.mutable_at(i, 0); // Assume row-major order
    }

    // Call the original function
    jpyo0803::SumByCol(in, result.mutable_data(), M, N);

    // Clean up
    delete[] in;

    return result;
}

template <typename T>
void shift_wrapper(py::array_t<T, py::array::c_style | py::array::forcecast> input, T amt) {
    if (input.ndim() != 3) {
        throw std::runtime_error("Input should be a 3D array");
    }

    auto buf = input.template mutable_unchecked<3>(); // Get buffer info for direct access
    int B = buf.shape(0);
    int M = buf.shape(1);
    int N = buf.shape(2);

    // Allocate T*** for input
    T*** in = new T**[B];
    for (int i = 0; i < B; ++i) {
        in[i] = new T*[M];
        for (int j = 0; j < M; ++j) {
            in[i][j] = &buf(i, j, 0);
        }
    }

    // Call the original function
    jpyo0803::Shift(in, amt, B, M, N);

    // Clean up
    for (int i = 0; i < B; ++i) {
        delete[] in[i];
    }
    delete[] in;
}


template <typename T>
void undo_shift_wrapper(py::array_t<T, py::array::c_style | py::array::forcecast> input,
                        T amt, T K,
                        py::array_t<T, py::array::c_style | py::array::forcecast> row_sum_x,
                        py::array_t<T, py::array::c_style | py::array::forcecast> col_sum_y) {
    // Check the input dimensions
    if (input.ndim() != 3 || row_sum_x.ndim() != 2 || col_sum_y.ndim() != 2) {
        throw std::runtime_error("Input should be a 3D array and sums should be 2D arrays");
    }

    auto buf = input.template mutable_unchecked<3>(); // 3D buffer
    auto r_sum = row_sum_x.template mutable_unchecked<2>(); // 2D row sums
    auto c_sum = col_sum_y.template mutable_unchecked<2>(); // 2D column sums
    int B = buf.shape(0);
    int M = buf.shape(1);
    int N = buf.shape(2);

    // Creating a pointer array to pass to UndoShift
    T*** in = new T**[B];
    for (int i = 0; i < B; ++i) {
        in[i] = new T*[M];
        for (int j = 0; j < M; ++j) {
            in[i][j] = &buf(i, j, 0);
        }
    }

    // Convert row_sum_x and col_sum_y to T** pointers
    T** r_sum_ptrs = new T*[B];
    T** c_sum_ptrs = new T*[B];
    for (int i = 0; i < B; ++i) {
        r_sum_ptrs[i] = &r_sum(i, 0);
        c_sum_ptrs[i] = &c_sum(i, 0);
    }

    // Call the function
    jpyo0803::UndoShift(in, amt, K, r_sum_ptrs, c_sum_ptrs, B, M, N);

    // Cleanup
    for (int i = 0; i < B; ++i) {
        delete[] in[i];
    }
    delete[] in;
    delete[] r_sum_ptrs;
    delete[] c_sum_ptrs;
}

template <typename T>
void tensor_add_wrapper(py::array_t<T, py::array::c_style | py::array::forcecast> input1,
                        py::array_t<T, py::array::c_style | py::array::forcecast> input2) {
    // Check the input dimensions
    if (input1.ndim() != 3 || input2.ndim() != 3) {
        throw std::runtime_error("Both inputs should be 3D arrays");
    }

    auto buf1 = input1.template mutable_unchecked<3>(); // 3D buffer for input1
    auto buf2 = input2.template mutable_unchecked<3>(); // 3D buffer for input2
    int B = buf1.shape(0);
    int M = buf1.shape(1);
    int N = buf1.shape(2);

    // Creating pointer arrays for both inputs
    T*** in1 = new T**[B];
    T*** in2 = new T**[B];
    for (int i = 0; i < B; ++i) {
        in1[i] = new T*[M];
        in2[i] = new T*[M];
        for (int j = 0; j < M; ++j) {
            in1[i][j] = &buf1(i, j, 0);
            in2[i][j] = &buf2(i, j, 0);
        }
    }

    // Call the TensorAdd function
    jpyo0803::TensorAdd(in1, in2, B, M, N);

    // Cleanup
    for (int i = 0; i < B; ++i) {
        delete[] in1[i];
        delete[] in2[i];
    }
    delete[] in1;
    delete[] in2;
}

py::list as_type_s8s32_wrapper(py::array_t<char, py::array::c_style | py::array::forcecast> input) {
    if (input.ndim() != 3) {
        throw std::runtime_error("Input should be a 3D array");
    }

    int B = input.shape(0);
    int M = input.shape(1);
    int N = input.shape(2);

    auto buf_input = input.mutable_unchecked<3>(); // Access the buffer

    // Create a 3D array pointer to pass to AsTypeS8S32
    char*** input_ptrs = new char**[B];
    for (int i = 0; i < B; ++i) {
        input_ptrs[i] = new char*[M];
        for (int j = 0; j < M; ++j) {
            input_ptrs[i][j] = &buf_input(i, j, 0);
        }
    }

    // Call AsTypeS8S32
    std::vector<std::vector<std::vector<int32_t>>> result = jpyo0803::AsTypeS8S32(input_ptrs, B, M, N);

    // Cleanup dynamically allocated memory
    for (int i = 0; i < B; ++i) {
        delete[] input_ptrs[i];
    }
    delete[] input_ptrs;

    // Convert the result to a Python list of lists
    py::list py_result;
    for (const auto& mat : result) {
        py::list py_mat;
        for (const auto& row : mat) {
            py_mat.append(py::cast(row));
        }
        py_result.append(py_mat);
    }

    return py_result;
}

void encrypt_tensor_light_wrapper(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tensor,
                                  uint32_t a, py::array_t<uint32_t, py::array::c_style | py::array::forcecast> b,
                                  bool vertical) {
    // Ensure tensor is a 3D array
    if (tensor.ndim() != 3) {
        throw std::runtime_error("Tensor should be a 3D array");
    }

    int B = tensor.shape(0);
    int M = tensor.shape(1);
    int N = tensor.shape(2);

    // Ensure that the size of `b` is consistent with the direction of encryption
    if ((vertical && b.size() != M) || (!vertical && b.size() != N)) {
        throw std::runtime_error("Incorrect size for b array");
    }

    // Accessing the mutable data pointers directly
    auto buf_tensor = tensor.mutable_unchecked<3>(); // 3D tensor buffer
    auto b_data = b.mutable_data(); // Direct access to b data

    // Create a pointer array to pass to EncryptTensorLight
    uint32_t*** tensor_ptrs = new uint32_t**[B];
    for (int i = 0; i < B; ++i) {
        tensor_ptrs[i] = new uint32_t*[M];
        for (int j = 0; j < M; ++j) {
            tensor_ptrs[i][j] = &buf_tensor(i, j, 0);
        }
    }

    // Call EncryptTensorLight with proper arguments
    jpyo0803::EncryptTensorLight(tensor_ptrs, a, b_data, vertical, B, M, N);

    // Cleanup dynamically allocated memory
    for (int i = 0; i < B; ++i) {
        delete[] tensor_ptrs[i];
    }
    delete[] tensor_ptrs;
}

void decrypt_tensor_light_wrapper(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tensor,
                                  uint32_t key_inv,
                                  py::array_t<uint32_t, py::array::c_style | py::array::forcecast> dec_row_sum_x,
                                  py::array_t<uint32_t, py::array::c_style | py::array::forcecast> dec_col_sum_y,
                                  uint32_t b_factor) {
    // Ensure tensor is a 3D array and sums are 2D arrays
    if (tensor.ndim() != 3 || dec_row_sum_x.ndim() != 2 || dec_col_sum_y.ndim() != 2) {
        throw std::runtime_error("Tensor should be a 3D array and dec_row_sum_x/dec_col_sum_y should be 2D arrays");
    }

    int B = tensor.shape(0);
    int M = tensor.shape(1);
    int N = tensor.shape(2);

    if (dec_row_sum_x.shape(0) != B || dec_row_sum_x.shape(1) != M || dec_col_sum_y.shape(0) != B || dec_col_sum_y.shape(1) != N) {
        throw std::runtime_error("Incorrect dimensions for dec_row_sum_x or dec_col_sum_y");
    }

    auto buf_tensor = tensor.mutable_unchecked<3>(); // 3D tensor buffer
    auto buf_dec_row = dec_row_sum_x.mutable_unchecked<2>(); // 2D dec_row_sum_x buffer
    auto buf_dec_col = dec_col_sum_y.mutable_unchecked<2>(); // 2D dec_col_sum_y buffer

    // Create pointer arrays to pass to DecryptTensorLight
    uint32_t*** tensor_ptrs = new uint32_t**[B];
    for (int i = 0; i < B; ++i) {
        tensor_ptrs[i] = new uint32_t*[M];
        for (int j = 0; j < M; ++j) {
            tensor_ptrs[i][j] = &buf_tensor(i, j, 0);
        }
    }

    uint32_t** dec_row_ptrs = new uint32_t*[B];
    for (int i = 0; i < B; ++i) {
        dec_row_ptrs[i] = &buf_dec_row(i, 0);
    }

    uint32_t** dec_col_ptrs = new uint32_t*[B];
    for (int i = 0; i < B; ++i) {
        dec_col_ptrs[i] = &buf_dec_col(i, 0);
    }

    // Call DecryptTensorLight with proper arguments
    jpyo0803::DecryptTensorLight(tensor_ptrs, key_inv, dec_row_ptrs, dec_col_ptrs, b_factor, B, M, N);

    // Cleanup dynamically allocated memory
    for (int i = 0; i < B; ++i) {
        delete[] tensor_ptrs[i];
    }
    delete[] tensor_ptrs;

    delete[] dec_row_ptrs;
    delete[] dec_col_ptrs;
}

template<typename T>
void bind_sum_by_row(py::module_& m, const std::string& typestr) {
    m.def(("SumByRow_" + typestr).c_str(), &sum_by_row_wrapper<T>,
          py::arg("input"),
          ("Sum elements of each row for type " + typestr).c_str());
}

template<typename T>
void bind_sum_by_col(py::module_& m, const std::string& typestr) {
    m.def(("SumByCol_" + typestr).c_str(), &sum_by_col_wrapper<T>,
          py::arg("input"),
          ("Sum elements of each col for type " + typestr).c_str());
}

template<typename T>
void bind_shift(py::module_& m, const std::string& typestr) {
    m.def(("Shift_" + typestr).c_str(), &shift_wrapper<T>,
          py::arg("input"), py::arg("amt"),
          ("Shift elements of a 3D array by a specified amount for type " + typestr).c_str());
}

template<typename T>
void bind_undo_shift(py::module_& m, const std::string& typestr) {
    m.def(("UndoShift_" + typestr).c_str(), &undo_shift_wrapper<T>,
          py::arg("input"), py::arg("amt"), py::arg("K"), py::arg("row_sum_x"), py::arg("col_sum_y"),
          ("Undo shift operation on a 3D array with control over row and column adjustments for type " + typestr).c_str());
}

template<typename T>
void bind_matrix_multiply(py::module_& m, const std::string& typestr) {
    m.def(("Matmul_" + typestr).c_str(), &jpyo0803::Matmul<T>,
          py::arg("X"), py::arg("Y"),
          ("Multiply two matrices of type " + typestr + " and return the result").c_str());
}

template <typename T>
void bind_generate_random_number(py::module_& m, const std::string& typestr) {
    m.def(("GenerateRandomNumber_" + typestr).c_str(), &jpyo0803::GenerateRandomNumber<T>,
          py::arg("low"), py::arg("high"),
          ("Generate a random number within a specified range for type " + typestr).c_str());
}

template <typename T>
void bind_randint_2d(py::module_& m, const std::string& typestr) {
    m.def(("RandInt2D_" + typestr).c_str(), &jpyo0803::RandInt2D<T>,
          py::arg("low"), py::arg("high"),py::arg("M"), py::arg("N"),
          ("Generate 2D random number array within a specified range for type " + typestr).c_str());
}

template<typename T>
void bind_tensor_add(py::module_& m, const std::string& typestr) {
    m.def(("TensorAdd_" + typestr).c_str(), &tensor_add_wrapper<T>,
          py::arg("input1"), py::arg("input2"),
          ("Add two 3D tensors element-wise for type " + typestr).c_str());
}

PYBIND11_MODULE(cipher_cpp, m) {
  m.doc() = "Test";
  bind_sum_by_row<int32_t>(m, "int32");
  bind_sum_by_row<int64_t>(m, "int64");
  bind_sum_by_row<uint32_t>(m, "uint32");
  bind_sum_by_row<uint64_t>(m, "uint64");
  
  bind_sum_by_col<int32_t>(m, "int32");
  bind_sum_by_col<int64_t>(m, "int64");
  bind_sum_by_col<uint32_t>(m, "uint32");
  bind_sum_by_col<uint64_t>(m, "uint64");
  
  bind_shift<int32_t>(m, "int32");
  bind_shift<int64_t>(m, "int64");
  bind_shift<uint32_t>(m, "uint32");
  bind_shift<uint64_t>(m, "uint64");
  
  bind_undo_shift<int32_t>(m, "int32");
  bind_undo_shift<int64_t>(m, "int64");
  bind_undo_shift<uint32_t>(m, "uint32");
  bind_undo_shift<uint64_t>(m, "uint64");
  
  bind_matrix_multiply<int32_t>(m, "int32");
  bind_matrix_multiply<int64_t>(m, "int64");
  bind_matrix_multiply<uint32_t>(m, "uint32");
  bind_matrix_multiply<uint64_t>(m, "uint64");

  bind_generate_random_number<int32_t>(m, "int32");
  bind_generate_random_number<int64_t>(m, "int64");
  bind_generate_random_number<uint32_t>(m, "uint32");
  bind_generate_random_number<uint64_t>(m, "uint64");

  bind_randint_2d<int32_t>(m, "int32");
  bind_randint_2d<int64_t>(m, "int64");
  bind_randint_2d<uint32_t>(m, "uint32");
  bind_randint_2d<uint64_t>(m, "uint64");

  bind_tensor_add<int32_t>(m, "int32");
  bind_tensor_add<uint32_t>(m, "uint32");
  bind_tensor_add<int64_t>(m, "int64");
  bind_tensor_add<uint64_t>(m, "uint64");

  m.def("AsTypeS8S32", &as_type_s8s32_wrapper,
          py::arg("input"),
          "Convert a 3D array from char to int32");

  m.def("EncryptTensorLight", &encrypt_tensor_light_wrapper,
          py::arg("tensor"), py::arg("a"), py::arg("b"), py::arg("vertical"),
          "Encrypt a tensor using specified parameters");

  m.def("DecryptTensorLight", &decrypt_tensor_light_wrapper,
          py::arg("tensor"), py::arg("key_inv"), py::arg("dec_row_sum_x"), py::arg("dec_col_sum_y"), py::arg("b_factor"),
          "Decrypt a tensor using specified parameters");

  m.def("EncryptMatrix2D", &jpyo0803::EncryptMatrix2D,
        "EncryptMatrix2D",
     py::arg("input"), py::arg("a"), py::arg("b"), py::arg("m"), py::arg("vertical"));

  m.def("EncryptMatrix2DFull", &jpyo0803::EncryptMatrix2DFull,
        "EncryptMatrix2DFull",
     py::arg("input"), py::arg("a"), py::arg("b"), py::arg("vertical"));

  m.def("DecryptMatrix2D", &jpyo0803::DecryptMatrix2D,
        "DecryptMatrix2D",
     py::arg("input"), py::arg("K"), py::arg("a1"), py::arg("b1"), py::arg("a2"), py::arg("b2"), py::arg("m"), py::arg("row_sum_x"), py::arg("col_sum_y"));
  
  m.def("DecryptMatrix2DFull", &jpyo0803::DecryptMatrix2DFull,
        "DecryptMatrix2DFull",
     py::arg("input"), py::arg("key_inv"), py::arg("dec_row_sum_x"), py::arg("dec_col_sum_y"), py::arg("b_factor"));

  m.def("GenerateKeySetA", &jpyo0803::GenerateKeySetA,
        "GenerateKeySetA",
     py::arg("mod"), py::arg("N"));

  m.def("FindKeyInvModFull", &jpyo0803::FindKeyInvModFull,
        "FindKeyInvModFull",
     py::arg("a_x"), py::arg("a_y"));

}
