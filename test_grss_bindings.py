#!/usr/bin/env python3
"""Regression tests for the standalone GRSS C++ -> Python bindings.

The test suite has two layers:

1. Exhaustive *coverage*: audit_grss_bindings.py inventories every GRSS C++
   function definition (excluding the existing src/grss.cpp binding layer) and
   checks that a Python callable exists for every unique function/method name,
   including internal file-local helpers and explicit overload aliases.

2. Numerical/object regression tests: exercise deterministic C++ routines,
   overload pairs, orbital-element round trips, matrix/vector operations,
   STM helpers, constructors, and PropSimulation object methods. Functions that
   intrinsically require an external SPICE kernel or a fully populated
   observational fixture are identified explicitly rather than being reported
   as passing without their required data.

Usage:
    python3 test_grss_bindings.py
"""
from __future__ import annotations

import importlib.util
import math
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "example_grss_full_bindings.py"
AUDIT = ROOT / "audit_grss_bindings.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(a: Any, b: Any, tol: float = 1e-10) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def assert_close(a: Any, b: Any, msg: str = "values differ", tol: float = 1e-10) -> None:
    if not close(a, b, tol):
        raise AssertionError(f"{msg}:\n  got      = {a!r}\n  expected = {b!r}")


def assert_finite(x: Any, label: str) -> None:
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            raise AssertionError(f"{label} is not finite: {x!r}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            assert_finite(v, f"{label}[{i}]")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_case(name: str, fn: Callable[[], None], counters: dict[str, int]) -> None:
    try:
        fn()
    except Exception as exc:
        counters["failed"] += 1
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        raise
    else:
        counters["passed"] += 1
        print(f"PASS  {name}")


def test_utilities(g):
    run = []
    run.append(("wrap_to_2pi", lambda: assert_close(g.wrap_to_2pi(-0.5), 2 * math.pi - 0.5)))
    run.append(("rad_to_deg return/out equivalence", lambda: assert_close(g.rad_to_deg(math.pi), g.rad_to_deg_out(math.pi))))
    run.append(("deg_to_rad return/out equivalence", lambda: assert_close(g.deg_to_rad(180), g.deg_to_rad_out(180))))
    run.append(("sort_vector ascending", lambda: expect(g.sort_vector([3.0, 1.0, 2.0], True) == ([1.0, 2.0, 3.0], [1, 2, 0]), "sort mismatch")))
    run.append(("sort_vector descending", lambda: expect(g.sort_vector([3.0, 1.0, 2.0], False) == ([3.0, 2.0, 1.0], [0, 2, 1]), "sort mismatch")))
    run.append(("sort_vector_by_idx", lambda: expect(g.sort_vector_by_idx([[30.0], [10.0], [20.0]], [1, 2, 0]) == [[10.0], [20.0], [30.0]], "index sort mismatch")))

    v1, v2 = [1.0, -2.0, 3.0], [4.0, 5.0, -6.0]
    run += [
        ("vdot", lambda: assert_close(g.vdot(v1, v2), -24.0)),
        ("vdot_raw", lambda: assert_close(g.vdot_raw(v1, v2, 3), -24.0)),
        ("vnorm", lambda: assert_close(g.vnorm(v1), math.sqrt(14.0))),
        ("vnorm_raw", lambda: assert_close(g.vnorm_raw(v1, 3), math.sqrt(14.0))),
        ("vunit", lambda: assert_close(g.vunit(v1), [x / math.sqrt(14.0) for x in v1])),
        ("vunit_raw", lambda: assert_close(g.vunit_raw(v1, 3), [x / math.sqrt(14.0) for x in v1])),
        ("vcross", lambda: assert_close(g.vcross(v1, v2), [-3.0, 18.0, 13.0])),
        ("vcross_raw", lambda: assert_close(g.vcross_raw(v1, v2), [-3.0, 18.0, 13.0])),
        ("vadd", lambda: assert_close(g.vadd(v1, v2), [5.0, 3.0, -3.0])),
        ("vsub", lambda: assert_close(g.vsub(v1, v2), [-3.0, -7.0, 9.0])),
        ("vcmul", lambda: assert_close(g.vcmul(v1, 2.5), [2.5, -5.0, 7.5])),
        ("vvmul", lambda: assert_close(g.vvmul(v1, v2), [4.0, -10.0, -18.0])),
        ("vabs_max", lambda: assert_close(g.vabs_max(v1), 3.0)),
        ("vabs_max_raw", lambda: assert_close(g.vabs_max_raw(v1, 3), 3.0)),
    ]
    for name, fn in run:
        yield name, fn


def test_matrices(g):
    A = [[2.0, 1.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    yield "mat_vec_mul", lambda: assert_close(g.mat_vec_mul(A, [2.0, 3.0]), [7.0, 18.0])
    yield "vec_mat_mul", lambda: assert_close(g.vec_mat_mul([2.0, 3.0], A), [13.0, 14.0])
    yield "mat_mat_mul", lambda: assert_close(g.mat_mat_mul(A, B), [[17.0, 20.0], [43.0, 50.0]])
    C = [[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]]
    C_inv_expected = [[13.0 / 9.0, -11.0 / 9.0, -5.0 / 9.0],
                      [-7.0 / 9.0, 8.0 / 9.0, 2.0 / 9.0],
                      [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0]]
    yield "mat3_inv", lambda: assert_close(g.mat3_inv(C), C_inv_expected, tol=1e-9)
    Cflat = [4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0]
    yield "mat3_mat3_mul", lambda: assert_close(g.mat3_mat3_mul(Cflat, Cflat), [x for row in g.mat_mat_mul(C, C) for x in row], tol=1e-9)
    yield "mat3_mat3_add", lambda: assert_close(g.mat3_mat3_add(Cflat, Cflat), [2 * x for x in Cflat], tol=1e-12)
    theta = 0.37
    for axis, fn in (("x", g.rot_mat_x), ("y", g.rot_mat_y), ("z", g.rot_mat_z)):
        R = fn(theta)
        yield f"rot_mat_{axis}", lambda R=R, axis=axis: _check_rotation(R, axis, theta)
    yield "LU_decompose/LU_inverse", lambda: _check_lu(g, A)
    yield "mat_inv", lambda: assert_close(g.mat_inv(A), [[0.8, -0.2], [-0.6, 0.4]], tol=1e-10)


def _check_rotation(R, axis, theta):
    c, s = math.cos(theta), math.sin(theta)
    if axis == "x":
        expected = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "y":
        expected = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        expected = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    assert_close(R, expected, tol=1e-12)


def _check_lu(g, A):
    LU, P = g.LU_decompose(A, 1e-16)
    _, invA = g.LU_inverse(LU, P)
    assert_close(invA, [[0.8, -0.2], [-0.6, 0.4]], tol=1e-10)


def test_time(g):
    jd = 2451545.0
    mjd = 51544.5
    et = 0.0
    pairs = [
        ("jd_to_et", g.jd_to_et(jd), et),
        ("jd_to_et_inplace", g.jd_to_et_inplace(jd), et),
        ("jd_to_mjd", g.jd_to_mjd(jd), mjd),
        ("jd_to_mjd_inplace", g.jd_to_mjd_inplace(jd), mjd),
        ("et_to_jd", g.et_to_jd(et), jd),
        ("et_to_jd_inplace", g.et_to_jd_inplace(et), jd),
        ("et_to_mjd", g.et_to_mjd(et), mjd),
        ("et_to_mjd_inplace", g.et_to_mjd_inplace(et), mjd),
        ("mjd_to_jd", g.mjd_to_jd(mjd), jd),
        ("mjd_to_jd_inplace", g.mjd_to_jd_inplace(mjd), jd),
        ("mjd_to_et", g.mjd_to_et(mjd), et),
        ("mjd_to_et_inplace", g.mjd_to_et_inplace(mjd), et),
    ]
    for name, got, expected in pairs:
        yield name, lambda got=got, expected=expected, name=name: assert_close(got, expected, name)
    yield "delta_at_utc", lambda: assert_close(g.delta_at_utc(58000.0), 37.0, tol=1e-12)
    yield "delta_at_tai finite", lambda: assert_finite(g.delta_at_tai(58000.0), "delta_at_tai")
    yield "delta_et_utc finite", lambda: assert_finite(g.delta_et_utc(58000.0), "delta_et_utc")
    yield "delta_et_tdb finite", lambda: assert_finite(g.delta_et_tdb(58000.0), "delta_et_tdb")


def test_elements(g):
    epoch = 60000.0
    kep = [2.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    comet = g.keplerian_to_cometary(epoch, kep)
    cart = g.keplerian_to_cartesian(kep)
    yield "keplerian_to_cometary", lambda: (expect(len(comet) == 6, "bad cometary length"), assert_finite(comet, "cometary"))
    yield "cometary_to_keplerian roundtrip", lambda: assert_close(g.cometary_to_keplerian(epoch, comet), kep, tol=1e-8)
    yield "keplerian_to_cartesian", lambda: (expect(len(cart) == 6, "bad cartesian length"), assert_finite(cart, "cartesian"))
    yield "cartesian_to_keplerian roundtrip", lambda: assert_close(g.cartesian_to_keplerian(cart), kep, tol=1e-7)
    comet2 = g.cartesian_to_cometary(epoch, cart)
    yield "cartesian_to_cometary", lambda: assert_close(comet2, comet, tol=1e-7)
    cart2 = g.cometary_to_cartesian(epoch, comet)
    yield "cometary_to_cartesian", lambda: assert_close(cart2, cart, tol=1e-7)
    E = g.kepler_solve_elliptic(0.1, 0.1)
    yield "kepler_solve_elliptic residual", lambda: assert_close(E - 0.1 * math.sin(E), 0.1, tol=1e-12)
    EHyp = g.kepler_solve_hyperbolic(0.1, 1.5)
    yield "kepler_solve_hyperbolic residual", lambda: assert_close(1.5 * math.sinh(EHyp) - EHyp, 0.1, tol=1e-12)
    M, E2, nu = g.kepler_solve(epoch, comet)
    e = comet[0]
    yield "kepler_solve residual", lambda: (assert_finite((M, E2, nu), "kepler_solve"), assert_close(E2 - e * math.sin(E2), M, tol=1e-10))
    yield "get_elements_partials", lambda: _check_partials_shape(g.get_elements_partials(epoch, kep, "kep2cart"))
    yield "get_cartesian_partials", lambda: _check_partials_shape(g.get_cartesian_partials(epoch, cart, "com2cart"))


def _check_partials_shape(x):
    expect(len(x) == 6, "partials must have 6 rows")
    expect(all(len(row) == 6 for row in x), "partials must be 6x6")
    assert_finite(x, "partials")


def test_misc_cpp(g):
    sim = g.PropSimulation("binding_test", 51544.5, 0, "")
    sim_copy = g.PropSimulation("binding_test_copy", sim)
    yield "PropSimulation copy constructor", lambda: expect(sim_copy.name == "binding_test_copy", "copy constructor failed")
    body = g.Body()
    body.set_J2(1.0e-3, 0.0, 90.0)
    yield "Body.set_J2", lambda: (assert_close(body.J2, 1.0e-3), assert_close(body.poleDec, math.pi / 2, tol=1e-12))

    ng = g.NongravParameters()
    pos, vel = [1.0, 0.0, 0.0], [0.0, 0.01, 0.0]
    ib_pos = g.IntegBody("test", 51544.5, 0.0, 1.0, pos, vel, ng)
    ib_comet = g.IntegBody("test_comet", 51544.5, 0.0, 1.0, [0.1, 0.9, 51544.5, 0.2, 0.3, 0.4], ng)
    yield "IntegBody cartesian constructor", lambda: expect(not ib_pos.isCometary, "cartesian constructor flag incorrect")
    yield "IntegBody cometary constructor", lambda: expect(ib_comet.isCometary, "cometary constructor flag incorrect")
    ib_pos.prepare_stm()
    yield "IntegBody.prepare_stm", lambda: expect(len(ib_pos.stm) == 36, "STM length incorrect")

    sb = g.SpiceBody("earth", 399, 51544.5, 0.0, 1.0)
    yield "SpiceBody constructor", lambda: expect(sb.spiceId == 399 and sb.isSpice, "SpiceBody fields incorrect")

    ev = g.Event()
    ev.bodyName = "test"
    ev.deltaV = [0.001, 0.002, 0.003]
    yield "Event constructor/fields", lambda: expect(ev.bodyName == "test" and ev.deltaV == [0.001, 0.002, 0.003], "Event fields incorrect")

    # Force routines require a fully initialized PropSimulation (and some
    # require populated STM/body/event state). They are covered by the
    # exhaustive binding-surface audit and the C++ oracle where deterministic,
    # rather than being invoked with an invalid empty fixture that could crash
    # inside the original C++ implementation.

    yield "get_atm_offset", lambda: assert_finite(g.get_atm_offset(399), "get_atm_offset")
    yield "rec_to_geodetic", lambda: (lambda r: (assert_close(r[0], 0.0, tol=1e-12), assert_finite(r, "rec_to_geodetic")))(g.rec_to_geodetic(1.0, 0.0, 0.0))
    yield "associated_legendre_function", lambda: _check_legendre(g.associated_legendre_function(0.2, 3))
    yield "root7", lambda: assert_close(g.root7(128.0), 2.0, tol=1e-12)
    yield "get_baseBodyFrame", lambda: expect(g.get_baseBodyFrame(399, 51544.5) == "ITRF93", "Earth base frame mismatch")

    sim.set_sim_constants()
    sim.set_integration_parameters(51545.5)
    yield "PropSimulation setters/getters", lambda: (expect(len(sim.get_sim_constants()) == 7, "Constants getter returned wrong size"), expect(len(sim.get_integration_parameters()) == 11, "IntegrationParameters getter returned wrong size"))

    sim.add_integ_body(ib_pos)
    yield "PropSimulation.add_integ_body", lambda: expect(sim.integBodies[0].name == "test", "integrated body not added")
    yield "PropSimulation.remove_body", lambda: (sim.remove_body("test"), expect(len(sim.integBodies) == 0, "integrated body not removed"))


def test_all_class_methods(g):
    """Exercise every non-kernel-dependent class method and explicitly classify
    methods whose execution needs external SPICE/PCK/observation fixtures.
    The source audit separately verifies that the classified methods exist.
    """
    # Body methods
    body = g.Body()
    body.set_J2(1.0e-3, 0.0, 90.0)
    expect(body.isJ2 and math.isclose(body.poleDec, math.pi / 2, abs_tol=1e-12),
           "Body.set_J2 did not update state")
    body = g.Body()
    body.set_harmonics(0.0, 90.0, 1, 1, [0.0, 0.0], [[0.0]], [[0.0]])
    expect(body.isHarmonic and body.nZon == 1 and body.nTes == 1,
           "Body.set_harmonics did not update state")

    # IntegBody method
    ng = g.NongravParameters()
    ib = g.IntegBody("method_test", 51544.5, 0.0, 1.0, [1,0,0], [0,0.01,0], ng)
    ib.prepare_stm()
    expect(len(ib.stm) == 36, "IntegBody.prepare_stm did not create the 36-element STM")

    # Event method: the binding explicitly returns the mutated vector& state.
    sim = g.PropSimulation("method_test", 51544.5, 0, "")
    sim.set_integration_parameters(51545.5)
    sim.add_integ_body(ib)
    ev = g.Event()
    ev.t = 51545.0
    ev.bodyName = "method_test"
    ev.bodyIndex = 0
    ev.xIntegIndex = 0
    ev.deltaV = [0.1, 0.2, 0.3]
    ev.multiplier = 1.0
    state = ev.apply_impulsive(sim, ev.t, [1,2,3,4,5,6])
    assert_close(state, [1,2,3,4.1,5.2,6.3], "Event.apply_impulsive")
    sim.remove_body("method_test")

    # Both print_summary overloads are side-effect-only, but safe on default objects.
    cap = g.CloseApproachParameters(); cap.print_summary()
    imp = g.ImpactParameters(); imp.print_summary()

    # PropSimulation methods that do not require loaded kernels.
    sim = g.PropSimulation("method_test", 51544.5, 0, "")
    sim.set_sim_constants()
    expect(len(sim.get_sim_constants()) == 7, "get_sim_constants returned wrong size")
    sim.set_integration_parameters(51545.5)
    expect(len(sim.get_integration_parameters()) == 11, "get_integration_parameters returned wrong size")
    # prepare_for_evaluation itself maps ephemeris even for an empty list, so
    # the audit checks its signature/presence rather than invoking it without kernels.

    spice = g.SpiceBody("earth", 399, 51544.5, 0.0, 1.0)
    sim.add_spice_body(spice)
    expect(len(sim.spiceBodies) == 1 and sim.integParams.nSpice == 1, "add_spice_body failed")
    try:
        sim.get_spiceBody_state(51544.5, "not_present")
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("get_spiceBody_state did not propagate the C++ missing-body error")
    sim.remove_body("earth")


def _check_legendre(P):
    expect(len(P) == 4 and all(len(row) == 4 for row in P), "Legendre shape incorrect")
    assert_finite(P, "Legendre")


def test_stm(g):
    yield "bcd_and_dot", lambda: _check_bcd(g.bcd_and_dot([1.0 if i % 7 == 0 else 0.0 for i in range(36)]))
    p = g.STMParameters(2)
    yield "STMParameters constructor", lambda: (expect(p.numParams == 2, "bad numParams"), expect(len(p.B) == 9 and len(p.D) == 6, "bad STM storage"))
    yield "stm_newton", lambda: _check_stm_result(g.stm_newton(p, 1.0, 1.0, 0.2, -0.3))
    yield "stm_ppn_simple", lambda: _check_stm_result(g.stm_ppn_simple(p, 1.0, 299792.458, 1.0, 1.0, 1.0, 0.2, -0.3, 0.01, -0.02, 0.03))
    yield "stm_J2", lambda: _check_stm_result(g.stm_J2(p, 1.0, 1e-3, 1.0, 0.2, -0.3, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0))
    ng = g.NongravParameters()
    yield "stm_nongrav", lambda: _check_stm_result(g.stm_nongrav(p, 1.0, ng, 1.0, 0.2, -0.3, 0.01, -0.02, 0.03, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    yield "bcd_2dot", lambda: _check_vector_result(g.bcd_2dot(p, 2, 0, [0.0] * 24))


def _check_bcd(value):
    expect(len(value) == 6 and all(len(v) == 9 for v in value), "bcd_and_dot returned wrong shape")
    assert_finite(value, "bcd_and_dot")


def _check_stm_result(value):
    # Most STM helpers return the 9 STM/partial arrays; stm_nongrav also returns
    # the updated rVec and nVec output vectors.
    expect(len(value) in (9, 11), "STM helper returned an unexpected tuple size")
    assert_finite(value, "STM helper")


def _check_vector_result(value):
    expect(len(value) == 24, "bcd_2dot result shape changed")
    assert_finite(value, "bcd_2dot")


def collect_regression_cases(g):
    counters = {"passed": 0, "failed": 0}
    for factory in (test_utilities, test_matrices, test_time, test_elements, test_misc_cpp, test_stm):
        for name, fn in factory(g):
            run_case(name, fn, counters)
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description="GRSS binding regression suite")
    parser.add_argument("--case", choices=[
        "test_utilities", "test_matrices", "test_time", "test_elements",
        "test_misc_cpp", "test_stm", "test_all_class_methods"
    ])
    args = parser.parse_args()

    if not EXAMPLE.exists() or not AUDIT.exists():
        print("This script expects example_grss_full_bindings.py and audit_grss_bindings.py in the GRSS repository root.", file=sys.stderr)
        return 2

    if args.case:
        # Case workers must not import the build helper module: it is unnecessary
        # here and can load additional state into the process before the C++
        # extension is exercised. The parent process already built the extension.
        # Case workers only import the already-built module; avoiding a second
        # build/load path keeps the isolated workers focused strictly on the
        # C++ calls being tested.
        # the already-built module; avoiding a second build/load path keeps the
        # isolated workers focused strictly on the C++ calls being tested.
        sys.path.insert(0, str(ROOT))
        import grss_full as g  # type: ignore
    else:
        example = load_module(EXAMPLE, "grss_binding_example")
        example.build_extension()
        sys.path.insert(0, str(ROOT))
        import grss_full as g  # type: ignore

    if args.case:
        counters = {"passed": 0, "failed": 0}
        if args.case == "test_all_class_methods":
            run_case("all non-kernel class-method smoke tests", lambda: test_all_class_methods(g), counters)
        else:
            for name, fn in getattr(sys.modules[__name__], args.case)(g):
                run_case(name, fn, counters)
        print(f"Case {args.case}: {counters['passed']} passed, {counters['failed']} failed")
        return 0 if counters["failed"] == 0 else 1

    print("Building/loading grss_full...")
    print("\n=== Exhaustive source/binding audit ===")
    audit = subprocess.run([sys.executable, str(AUDIT)], cwd=ROOT)
    if audit.returncode != 0:
        return audit.returncode

    # Run independent groups in fresh Python processes. This prevents a C++
    # routine with process-global state or an object-lifetime bug from corrupting
    # a later, unrelated regression group. Every group still exercises the actual
    # compiled grss_full extension.
    cases = [
        "test_utilities", "test_matrices", "test_time", "test_elements",
        "test_misc_cpp", "test_stm", "test_all_class_methods"
    ]
    print("\n=== Deterministic/object regression groups (isolated processes) ===")
    failed = []
    for case in cases:
        print(f"\n--- {case} ---")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--case", case], cwd=ROOT)
        if result.returncode != 0:
            failed.append(case)

    advertised = tuple(getattr(g, "__cpp_functions__", ()))
    print("\n=== Summary ===")
    print(f"Unique C++ callable names audited: {g.__cpp_unique_callable_count__}")
    print(f"Source function definitions audited: {g.__cpp_source_definition_count__}")
    print(f"Binding entry points advertised in module metadata: {len(advertised)}")
    print(f"Regression groups passed: {len(cases) - len(failed)} / {len(cases)}")
    print(f"Regression groups failed: {len(failed)}")

    print("\nKernel-dependent C++ functions are intentionally not reported as numerically tested here:")
    print("  PCK/SPK file loading, SPICE state lookup, full observation/light-time models, and")
    print("  propagation paths requiring initialized SPICE kernels or a populated observation fixture.")
    print("The exhaustive audit still requires a Python binding for every one of those functions.")

    if failed:
        print("\nFailed groups:")
        for case in failed:
            print(f"  - {case}")
        return 1

    print("\nGRSS binding regression suite: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
