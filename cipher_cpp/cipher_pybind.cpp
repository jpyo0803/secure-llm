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
    if (input.ndim() != 2) {
        throw std::runtime_error("Input should be a 2D array");
    }

    auto buf = input.template mutable_unchecked<2>(); // Get buffer info for direct access
    int M = buf.shape(0);
    int N = buf.shape(1);

    // Allocate int** for input
    T** in = new T*[M];
    for (int i = 0; i < M; ++i) {
        in[i] = &buf(i, 0);
    }

    // Call the original function
    jpyo0803::Shift(in, amt, M, N);

    // Clean up
    delete[] in;
}

template <typename T>
void undo_shift_wrapper(py::array_t<T, py::array::c_style | py::array::forcecast> input,
                              T amt, T K,
                              py::array_t<T, py::array::c_style | py::array::forcecast> row_sum_x,
                              py::array_t<T, py::array::c_style | py::array::forcecast> col_sum_y) {
    if (input.ndim() != 2 || row_sum_x.ndim() != 1 || col_sum_y.ndim() != 1) {
        throw std::runtime_error("Input and sums should be 2D and 1D arrays respectively");
    }

    auto buf = input.template mutable_unchecked<2>(); // Get buffer info for direct access
    auto r_sum = row_sum_x.template unchecked<1>(); // Direct access to row_sum_x elements
    auto c_sum = col_sum_y.template unchecked<1>(); // Direct access to col_sum_y elements
    int M = buf.shape(0);
    int N = buf.shape(1);

    T** in = new T*[M];
    for (int i = 0; i < M; ++i) {
        in[i] = &buf(i, 0);
    }

    T* row_sums = row_sum_x.mutable_data();
    T* col_sums = col_sum_y.mutable_data();

    jpyo0803::UndoShift(in, amt, K, row_sums, col_sums, M, N);

    delete[] in;
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
          ("Shift elements of a 2D array by a specified amount for type " + typestr).c_str());
}

template<typename T>
void bind_undo_shift(py::module_& m, const std::string& typestr) {
    m.def(("UndoShift_" + typestr).c_str(), &undo_shift_wrapper<T>,
          py::arg("input"), py::arg("amt"), py::arg("K"), py::arg("row_sum_x"), py::arg("col_sum_y"),
          ("Undo shift operation on a 2D array with control over row and column adjustments for type " + typestr).c_str());
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
