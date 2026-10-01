"""Numerical invariants and mechanism checks, run with python3 -m unittest."""
from dataclasses import replace
import unittest
import numpy as np
from model import Parameters, Simulation


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.p = Parameters(n=100, size=16, grid=32)

    def test_mass_and_periodicity(self):
        sim = Simulation(self.p)
        sim.xy[0] = [0, self.p.size-1e-8]
        before = sim.density()
        self.assertAlmostEqual(before.sum()*sim.dx**2, self.p.n, places=10)
        sim.xy = (sim.xy + sim.dx) % self.p.size
        np.testing.assert_allclose(sim.density(), np.roll(before, (1, 1), (0, 1)), atol=1e-13)
        sim.step(500)
        self.assertTrue(np.all((sim.xy >= 0) & (sim.xy < self.p.size)))

    def test_visual_bodies_do_not_change_dynamics(self):
        for mode in ("density", "oxygen"):
            p = replace(self.p, mode=mode)
            a, b = Simulation(p, trails=True), Simulation(p, trails=False)
            a.step(100)
            b.step(100)
            np.testing.assert_array_equal(a.xy, b.xy)
            np.testing.assert_array_equal(a.oxygen, b.oxygen)
            lengths = np.linalg.norm(np.diff(a.body, axis=1), axis=2)
            np.testing.assert_allclose(lengths, p.body_length/(p.body_points-1), atol=1e-12)
            self.assertTrue(np.all(np.abs(np.diff(a.segments(), axis=1)) < p.size/2))

    def test_uniform_oxygen_control(self):
        sim = Simulation(replace(self.p, mode="oxygen", uptake=0))
        sim.step(100)
        np.testing.assert_allclose(sim.oxygen, 1.0, atol=1e-13)

    def test_oxygen_reaction_equilibrium_and_bounds(self):
        sim = Simulation(replace(self.p, mode="oxygen"))
        sim.rho[:] = 3.0
        for _ in range(1500):
            sim._advance_oxygen()
        expected = sim.p.replenishment/(sim.p.replenishment+sim.p.uptake*3)
        np.testing.assert_allclose(sim.oxygen, expected, atol=1e-10)
        sim.step(100)
        self.assertGreaterEqual(sim.oxygen.min(), -1e-12)
        self.assertLessEqual(sim.oxygen.max(), 1+1e-12)

    def test_speed_response(self):
        sim = Simulation(self.p)
        self.assertTrue(np.all(np.diff(sim.speed_fraction(np.linspace(0, 10, 30))) < 0))
        sim = Simulation(replace(self.p, mode="oxygen"))
        preference = sim.p.preferred_oxygen
        self.assertAlmostEqual(sim.speed_fraction(preference), sim.p.min_speed)
        self.assertGreater(sim.speed_fraction(0), sim.speed_fraction(preference))
        self.assertGreater(sim.speed_fraction(1), sim.speed_fraction(preference))

    def test_turning_reverses_below_preferred_oxygen(self):
        p = replace(self.p, mode="oxygen", persistence=1e20, diffusion=0,
                    replenishment=0, uptake=0)
        for level, expected_sign in ((0.8, 1), (0.2, -1)):
            sim = Simulation(p)
            sim.xy[:] = [0, 8]
            sim.theta[:] = np.pi/2  # north; oxygen increases to the east
            x = np.arange(p.grid)*sim.dx
            sim.oxygen[:] = level + 0.05*np.sin(2*np.pi*x/p.size)
            sim.step()
            self.assertTrue(np.all((sim.theta-np.pi/2)*expected_sign > 0))

    def test_periodic_connected_component(self):
        sim = Simulation(self.p)
        sim.rho[:] = 0
        sim.rho[10, 0] = 5
        sim.rho[10, -1] = 5
        sim.rho[20, 20] = 5
        self.assertAlmostEqual(sim.metrics()["largest"], 2/3)

    def test_seed_reproducible(self):
        a, b = Simulation(self.p), Simulation(self.p)
        a.step(50)
        b.step(20)
        b.step(30)
        np.testing.assert_array_equal(a.xy, b.xy)
        self.assertEqual(a.time, b.time)


if __name__ == "__main__":
    unittest.main()
