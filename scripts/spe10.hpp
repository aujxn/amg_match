// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SPE10_HPP
#define MFEM_SPE10_HPP

#include "mfem.hpp"

namespace mfem {

struct SPE10Data {
  static constexpr int nx = 60;
  static constexpr int ny = 220;
  static constexpr int nz = 85;

  Array<double> coeff_data;
  SPE10Data() : coeff_data(3 * nx * ny * nz) {
    std::ifstream permfile("../spe10/spe_perm.dat");
    if (!permfile.good()) {
      MFEM_ABORT("Cannot open data file spe_perm.dat.")
    }
    double tmp;
    double *ptr = coeff_data.begin();
    for (int l = 0; l < 3; l++) {
      for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            permfile >> *ptr;
            //*ptr = 1.0 / (*ptr);
            ptr++;
          }
          for (int i = 0; i < 60 - nx; i++) {
            permfile >> tmp; // skip unneeded part
          }
        }
        for (int j = 0; j < 220 - ny; j++)
          for (int i = 0; i < 60; i++) {
            permfile >> tmp; // skip unneeded part
          }
      }
      if (l < 2) // if not processing Kz, we must skip unneeded part
      {
        for (int k = 0; k < 85 - nz; k++)
          for (int j = 0; j < 220; j++)
            for (int i = 0; i < 60; i++) {
              permfile >> tmp; // skip unneeded part
            }
      }
    }
  }
  void Invert() {
    for (int i = 0; i < coeff_data.Size(); ++i) {
      coeff_data[i] = 1.0 / coeff_data[i];
    }
  }
};

struct SPE10Coefficient : DiagonalMatrixCoefficient {
  SPE10Data spe10;
  SPE10Coefficient() : VectorCoefficient(3) {}
  using VectorCoefficient::Eval;
  void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) {
    constexpr int nx = SPE10Data::nx;
    constexpr int ny = SPE10Data::ny;
    constexpr int nz = SPE10Data::nz;
    double data[3];
    Vector xvec(data, 3);
    T.Transform(ip, xvec);

    const double x = xvec[0], y = xvec[1], z = xvec[2];
    const int i = (nx * x) / 1200;
    MFEM_VERIFY(i >= 0 && i < nx, "");
    const int j = (ny * y) / 2200;
    MFEM_VERIFY(j >= 0 && j < ny, "");
    const int k = (nz * z) / 170;
    MFEM_VERIFY(k >= 0 && k < nz, "");

    V[0] = spe10.coeff_data[ny * nx * k + nx * j + i];
    V[1] = spe10.coeff_data[ny * nx * k + nx * j + i + nx * ny * nz];
    V[2] = spe10.coeff_data[ny * nx * k + nx * j + i + 2 * nx * ny * nz];

    // std::cout << x << '\t' << y << '\t' << z << '\n';
    // std::cout << i << '\t' << j << '\t' << k << '\t' << V[0] << '\t' << V[1]
    // << '\t' << V[2] << '\n';
  }
};

struct ScalarSPE10Coefficient : Coefficient {
  SPE10Coefficient vector_coeff;
  Vector val = Vector(3);
  using Coefficient::Eval;
  double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    vector_coeff.Eval(val, T, ip);
    // std::cout << val.Norml2() << '\n';
    const double nrm = val.Norml2();
    MFEM_VERIFY(nrm >= 0, "");
    return nrm;
  }
};

} // namespace mfem

#endif
