#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cipher.h"
#include <iostream>
#include <stdint.h>

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

template<typename T>
void bind_sum_by_row(py::module_& m, const std::string& typestr) {
    m.def(("SumByRow_" + typestr).c_str(), &sum_by_row_wrapper<T>,
          py::arg("input"),
          ("Sum elements of each row for type " + typestr).c_str());
}

PYBIND11_MODULE(cipher_cpp, m) {
  m.doc() = "Test";
  bind_sum_by_row<int32_t>(m, "int32");
  bind_sum_by_row<int64_t>(m, "int64");
  bind_sum_by_row<uint32_t>(m, "uint32");
  bind_sum_by_row<uint64_t>(m, "uint64");
}
