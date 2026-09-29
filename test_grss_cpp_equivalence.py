#!/usr/bin/env python3
"""Compare direct C++ GRSS calls with the Python/pybind11 calls.

This test deliberately compiles a tiny C++ oracle against the original GRSS
C++ implementation (excluding the repository's existing Python binding file),
then compares its results with calls through grss_full.

It complements audit_grss_bindings.py: the audit proves coverage, while this
script proves that the Python wrapper returns the same values for deterministic
functions and selected stateful methods.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "example_grss_full_bindings.py"
ORACLE_CPP = ROOT / ".grss_binding_oracle.cpp"
ORACLE_EXE = ROOT / ".grss_binding_oracle"

ORACLE_SOURCE = r'''
#define private public
#define protected public
#include "src/approach.cpp"
#include "src/elements.cpp"
#include "src/force.cpp"
#include "src/ias15.cpp"
#include "src/interpolate.cpp"
#include "src/observe.cpp"
#include "src/parallel.cpp"
#include "src/simulation.cpp"
#define _mjd pck_mjd_internal
#include "src/pck.cpp"
#undef _mjd
#define _mjd spk_mjd_internal
#include "src/spk.cpp"
#undef _mjd
#include "src/stm.cpp"
#include "src/timeconvert.cpp"
#include "src/utilities.cpp"
#include "grss.h"
#undef protected
#undef private
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static void put_scalar(const char* k, long double v) {
    std::cout << k << "|S|" << std::setprecision(18) << (double)v << "\n";
}
static void put_vec(const char* k, const std::vector<real>& v) {
    std::cout << k << "|V|";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << std::setprecision(18) << (double)v[i];
    }
    std::cout << "\n";
}
static void put_mat(const char* k, const std::vector<std::vector<real>>& a) {
    std::cout << k << "|M|";
    for (size_t i = 0; i < a.size(); ++i) {
        if (i) std::cout << ";";
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (j) std::cout << ",";
            std::cout << std::setprecision(18) << (double)a[i][j];
        }
    }
    std::cout << "\n";
}

int main() {
    std::cout << "ORACLE_VERSION|S|1\n";

    real x = -0.5; wrap_to_2pi(x); put_scalar("wrap_to_2pi", x);
    const real PIc = 3.1415926535897932384626433832795L;
    real y = 0; rad_to_deg((real)PIc, y); put_scalar("rad_to_deg", y);
    real z = 0; deg_to_rad((real)180.0, z); put_scalar("deg_to_rad", z);

    std::vector<real> v1 = {1.0, -2.0, 3.0};
    std::vector<real> v2 = {4.0, 5.0, -6.0};
    real out = 0;
    vdot(v1, v2, out); put_scalar("vdot", out);
    vnorm(v1, out); put_scalar("vnorm", out);
    std::vector<real> vu(3); vunit(v1, vu); put_vec("vunit", vu);
    std::vector<real> vc(3); vcross(v1.data(), v2.data(), vc.data()); put_vec("vcross", vc);
    std::vector<real> va(3); vcmul(v1, 2.5, va); put_vec("vcmul", va);
    std::vector<real> vv(3); vvmul(v1, v2, vv); put_vec("vvmul", vv);
    vabs_max(v1, out); put_scalar("vabs_max", out);

    std::vector<std::vector<real>> A = {{2,1},{3,4}};
    std::vector<std::vector<real>> B = {{5,6},{7,8}};
    std::vector<real> mv(2), vm(2);
    mat_vec_mul(A, {2,3}, mv); put_vec("mat_vec_mul", mv);
    real** Araw = new real*[2];
    for (size_t i = 0; i < 2; ++i) Araw[i] = new real[2];
    Araw[0][0]=2; Araw[0][1]=1; Araw[1][0]=3; Araw[1][1]=4;
    vec_mat_mul({2,3}, Araw, 2, vm); put_vec("vec_mat_mul", vm);
    delete[] Araw[0]; delete[] Araw[1]; delete[] Araw;
    std::vector<std::vector<real>> MM(2, std::vector<real>(2)); mat_mat_mul(A, B, MM); put_mat("mat_mat_mul", MM);
    std::vector<std::vector<real>> inv(2, std::vector<real>(2)); mat_inv(A, inv); put_mat("mat_inv", inv);

    std::vector<std::vector<real>> Rx(3, std::vector<real>(3)); rot_mat_x(0.37, Rx); put_mat("rot_mat_x", Rx);
    std::vector<std::vector<real>> Ry(3, std::vector<real>(3)); rot_mat_y(0.37, Ry); put_mat("rot_mat_y", Ry);
    std::vector<std::vector<real>> Rz(3, std::vector<real>(3)); rot_mat_z(0.37, Rz); put_mat("rot_mat_z", Rz);

    put_scalar("jd_to_mjd", jd_to_mjd(2451545.0));
    put_scalar("jd_to_et", jd_to_et(2451545.0));
    put_scalar("mjd_to_jd", mjd_to_jd(51544.5));
    put_scalar("mjd_to_et", mjd_to_et(51544.5));
    put_scalar("et_to_jd", et_to_jd(0.0));
    put_scalar("et_to_mjd", et_to_mjd(0.0));
    put_scalar("delta_at_utc", delta_at_utc(58000.0));

    std::vector<real> kep = {2.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<real> cart(6), comet(6);
    keplerian_to_cartesian(kep, cart); put_vec("keplerian_to_cartesian", cart);
    std::vector<real> kepRound(6); cartesian_to_keplerian(cart, kepRound); put_vec("cartesian_to_keplerian", kepRound);
    keplerian_to_cometary(60000.0, {2.0,0.1,0.2,0.3,0.4,0.5}, comet); put_vec("keplerian_to_cometary", comet);
    std::vector<real> kep2(6); cometary_to_keplerian(60000.0, comet, kep2); put_vec("cometary_to_keplerian", kep2);
    std::vector<real> cart2(6); cometary_to_cartesian(60000.0, comet, cart2); put_vec("cometary_to_cartesian", cart2);
    std::vector<real> comet2(6); cartesian_to_cometary(60000.0, cart, comet2); put_vec("cartesian_to_cometary", comet2);
    real E = 0; kepler_solve_elliptic(0.1, 0.1, E, 1.0e-12L, 250); put_scalar("kepler_solve_elliptic", E);
    real EH = 0; kepler_solve_hyperbolic(0.1, 1.5, EH, 1.0e-12L, 250); put_scalar("kepler_solve_hyperbolic", EH);

    std::vector<std::vector<real>> P(4, std::vector<real>(4)); associated_legendre_function(0.2, 3, P); put_mat("associated_legendre_function", P);
    put_scalar("root7", root7(128.0));
    std::string frame; get_baseBodyFrame(399, 51544.5, frame); std::cout << "get_baseBodyFrame|T|" << frame << "\n";

    Body body; body.set_J2(1.0e-3, 0.0, 90.0);
    put_scalar("Body.set_J2.J2", body.J2); put_scalar("Body.set_J2.poleDec", body.poleDec);

    PropSimulation sim("oracle", 51544.5, 0, "");
    sim.set_sim_constants();
    put_vec("PropSimulation.get_sim_constants", sim.get_sim_constants());
    sim.set_integration_parameters(51545.5);
    put_vec("PropSimulation.get_integration_parameters", sim.get_integration_parameters());

    NongravParameters ng;
    IntegBody ib("test", 51544.5, 0.0, 1.0, {1.0,0.0,0.0}, {0.0,0.01,0.0}, ng);
    sim.add_integ_body(ib);
    Event ev; ev.t = 51545.0; ev.bodyName = "test"; ev.bodyIndex = 0; ev.xIntegIndex = 0;
    ev.deltaV = {0.1,0.2,0.3}; ev.multiplier = 1.0;
    std::vector<real> state = {1,2,3,4,5,6};
    ev.apply_impulsive(&sim, ev.t, state); put_vec("Event.apply_impulsive", state);
    sim.remove_body("test");

    return 0;
}
'''


def parse_oracle(text: str):
    out = {}
    for line in text.splitlines():
        key, kind, value = line.split("|", 2)
        if kind == "S":
            out[key] = float(value)
        elif kind == "V":
            out[key] = [float(x) for x in value.split(",") if x]
        elif kind == "M":
            out[key] = [[float(x) for x in row.split(",")] for row in value.split(";")]
        elif kind == "T":
            out[key] = value
        else:
            raise RuntimeError(f"Unknown oracle kind {kind!r}")
    return out


def find_cxx():
    explicit = os.environ.get("CXX")
    candidates = [explicit] if explicit else []
    if sys.platform == "darwin":
        for v in range(20, 8, -1):
            candidates += [f"g++-{v}", f"gcc-{v}"]
    candidates += ["g++", "c++", "clang++"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if resolved and Path(resolved).exists():
            version = subprocess.check_output([resolved, "--version"], text=True, stderr=subprocess.STDOUT)
            if "Apple clang" not in version and "Apple LLVM" not in version:
                return resolved
    raise RuntimeError("Could not find a non-Apple-Clang C++ compiler. Set CXX to Homebrew GCC.")


def build_oracle(cxx: str):
    ORACLE_CPP.write_text(ORACLE_SOURCE)
    cmd = [cxx, "-std=c++11", "-O2", "-fopenmp", f"-I{ROOT / 'include'}", str(ORACLE_CPP)]
    if sys.platform == "darwin":
        cmd += ["-o", str(ORACLE_EXE)]
    else:
        cmd += ["-o", str(ORACLE_EXE)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def close(a, b, tol=2e-12):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def main():
    spec = __import__("importlib.util").util.spec_from_file_location("example", EXAMPLE)
    example = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.build_extension()
    sys.path.insert(0, str(ROOT))
    import grss_full as g

    cxx = find_cxx()
    build_oracle(cxx)
    oracle = parse_oracle(subprocess.check_output([str(ORACLE_EXE)], cwd=ROOT, text=True))

    py = {}
    py["wrap_to_2pi"] = (lambda x: x)(-0.5)
    py["wrap_to_2pi"] = g.wrap_to_2pi(-0.5)
    py["rad_to_deg"] = g.rad_to_deg(math.pi)
    py["deg_to_rad"] = g.deg_to_rad(180.0)
    py["vdot"] = g.vdot([1.0,-2.0,3.0],[4.0,5.0,-6.0])
    py["vnorm"] = g.vnorm([1.0,-2.0,3.0])
    py["vunit"] = g.vunit([1.0,-2.0,3.0])
    py["vcross"] = g.vcross([1.0,-2.0,3.0],[4.0,5.0,-6.0])
    py["vcmul"] = g.vcmul([1.0,-2.0,3.0],2.5)
    py["vvmul"] = g.vvmul([1.0,-2.0,3.0],[4.0,5.0,-6.0])
    py["vabs_max"] = g.vabs_max([1.0,-2.0,3.0])
    py["mat_vec_mul"] = g.mat_vec_mul([[2,1],[3,4]],[2,3])
    py["vec_mat_mul"] = g.vec_mat_mul([2,3],[[2,1],[3,4]])
    py["mat_mat_mul"] = g.mat_mat_mul([[2,1],[3,4]],[[5,6],[7,8]])
    py["mat_inv"] = g.mat_inv([[2,1],[3,4]])
    py["rot_mat_x"] = g.rot_mat_x(0.37)
    py["rot_mat_y"] = g.rot_mat_y(0.37)
    py["rot_mat_z"] = g.rot_mat_z(0.37)
    py["jd_to_mjd"] = g.jd_to_mjd(2451545.0)
    py["jd_to_et"] = g.jd_to_et(2451545.0)
    py["mjd_to_jd"] = g.mjd_to_jd(51544.5)
    py["mjd_to_et"] = g.mjd_to_et(51544.5)
    py["et_to_jd"] = g.et_to_jd(0.0)
    py["et_to_mjd"] = g.et_to_mjd(0.0)
    py["delta_at_utc"] = g.delta_at_utc(58000.0)
    kep = [2.0,0.1,0.2,0.3,0.4,0.5]
    py["keplerian_to_cartesian"] = g.keplerian_to_cartesian(kep)
    py["cartesian_to_keplerian"] = g.cartesian_to_keplerian(g.keplerian_to_cartesian(kep))
    py["keplerian_to_cometary"] = g.keplerian_to_cometary(60000.0,kep)
    py["cometary_to_keplerian"] = g.cometary_to_keplerian(60000.0,py["keplerian_to_cometary"])
    py["cometary_to_cartesian"] = g.cometary_to_cartesian(60000.0,py["keplerian_to_cometary"])
    py["cartesian_to_cometary"] = g.cartesian_to_cometary(60000.0,py["keplerian_to_cartesian"])
    py["kepler_solve_elliptic"] = g.kepler_solve_elliptic(0.1,0.1)
    py["kepler_solve_hyperbolic"] = g.kepler_solve_hyperbolic(0.1,1.5)
    py["associated_legendre_function"] = g.associated_legendre_function(0.2,3)
    py["root7"] = g.root7(128.0)
    py["get_baseBodyFrame"] = g.get_baseBodyFrame(399,51544.5)
    body = g.Body(); body.set_J2(1e-3,0.0,90.0)
    py["Body.set_J2.J2"] = body.J2
    py["Body.set_J2.poleDec"] = body.poleDec
    sim = g.PropSimulation("oracle",51544.5,0,"")
    sim.set_sim_constants(); py["PropSimulation.get_sim_constants"] = sim.get_sim_constants()
    sim.set_integration_parameters(51545.5); py["PropSimulation.get_integration_parameters"] = sim.get_integration_parameters()
    ng = g.NongravParameters(); ib = g.IntegBody("test",51544.5,0.0,1.0,[1.0,0.0,0.0],[0.0,0.01,0.0],ng); sim.add_integ_body(ib)
    ev = g.Event(); ev.t=51545.0; ev.bodyName="test"; ev.bodyIndex=0; ev.xIntegIndex=0; ev.deltaV=[0.1,0.2,0.3]; ev.multiplier=1.0
    state=[1,2,3,4,5,6]
    # The standalone binding converts the C++ vector& output to a returned Python list.
    py["Event.apply_impulsive"] = ev.apply_impulsive(sim,ev.t,state)

    failures=[]
    for k,v in oracle.items():
        if k == "ORACLE_VERSION": continue
        if k not in py:
            failures.append(f"Python side missing oracle case: {k}")
            continue
        if not close(py[k],v):
            failures.append(f"{k}: Python={py[k]!r} C++={v!r}")
        else:
            print(f"PASS  {k}")
    if failures:
        print("\nC++/Python equivalence failures:")
        for f in failures: print("  "+f)
        return 1
    print(f"\nC++/Python numerical/state equivalence: PASS ({len(oracle)-1} cases)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
