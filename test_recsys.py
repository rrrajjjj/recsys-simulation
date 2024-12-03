import unittest
import random
from simulation import Patient, Session, Protocol, RecSys


class TestRecSys(unittest.TestCase):
    def setUp(self):
        # Initialize protocols
        self.protocols = [
            Protocol(id=1, name="Protocol A", dms=["dm1", "dm2"], ar=True),
            Protocol(id=2, name="Protocol B", dms=["dm1", "dm2", "dm3"], ar=False),
        ]

        # Initialize RecSys
        self.recsys = RecSys(protocols=self.protocols)

        # Initialize a patient
        self.patient = Patient(dm_ground_truth={
            "1": [0.8, 0.9],
            "2": [0.7, 0.8, 0.9]
        })

        # Add sessions for protocol ID 1
        self.sessions_protocol_1 = [
            Session(session_id=1, protocol_id=1, dms=[0.1, 0.2], timestamp=1000),
            Session(session_id=2, protocol_id=1, dms=[0.3, 0.4], timestamp=2000),
        ]
        for session in self.sessions_protocol_1:
            self.patient.update(session)

        # Add a single session for protocol ID 2
        self.sessions_protocol_2 = [
            Session(session_id=1, protocol_id=2, dms=[0.5, 0.6, 0.7], timestamp=3000),
        ]
        for session in self.sessions_protocol_2:
            self.patient.update(session)

    def test_compute_delta_dm_multiple_sessions(self):
        sessions = self.patient.get_last_n_sessions(protocol_id=1, n=2)
        delta_dm = self.recsys.compute_delta_dm(sessions)
        expected_delta = abs(0.3 - 0.1) + abs(0.4 - 0.2)
        self.assertAlmostEqual(delta_dm, expected_delta, places=5)

    def test_compute_delta_dm_single_session(self):
        sessions = self.patient.get_last_n_sessions(protocol_id=2, n=2)
        delta_dm = self.recsys.compute_delta_dm(sessions)
        expected_delta = sum([0.5, 0.6, 0.7])
        self.assertAlmostEqual(delta_dm, expected_delta, places=5)

    def test_compute_delta_dm_no_sessions(self):
        random.seed(42)  # Set seed for reproducibility
        sessions = self.patient.get_last_n_sessions(protocol_id=3, n=2)
        delta_dm = self.recsys.compute_delta_dm(sessions)
        self.assertTrue(0.8 <= delta_dm <= 1.0)

    def test_recommendation_with_parameters(self):
        params = {"timestamp": 1700000000}
        recommended_protocols = self.recsys.recommend(self.patient, n=1, params=params)

        self.assertEqual(len(self.patient.recommendation_history), 1)
        history = self.patient.recommendation_history[0]
        self.assertIn("prob_dist", history)
        self.assertIn("recommended", history)
        self.assertIn("params", history)
        self.assertEqual(history["params"]["n"], 1)
        self.assertEqual(len(history["recommended"]), 1)
        self.assertIn(history["recommended"][0], [1, 2])
        self.assertIsInstance(recommended_protocols[0], Protocol)

    def test_recommendation_with_no_sessions(self):
        random.seed(42)
        new_patient = Patient(dm_ground_truth={
            "1": [0.8, 0.9],
            "2": [0.7, 0.8, 0.9]
        })
        params = {"strategy": "weighted_random_sampling"}
        recommended_protocols = self.recsys.recommend(new_patient, n=2, params=params)

        self.assertEqual(len(new_patient.recommendation_history), 1)
        history = new_patient.recommendation_history[0]
        self.assertIn("prob_dist", history)
        self.assertEqual(len(history["recommended"]), 2)
        prob_dist = history.get("prob_dist", {})  # Use .get() to safely access prob_dist

        self.assertAlmostEqual(prob_dist.get(1, None), 0.5, places=1)
        self.assertAlmostEqual(prob_dist.get(2, None), 0.5, places=1)

        self.assertEqual(len(recommended_protocols), 2)
        self.assertIsInstance(recommended_protocols[0], Protocol)

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()