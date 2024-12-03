import unittest
import time
from simulation import Patient, Session, Protocol  # Import classes from your module

class TestPatient(unittest.TestCase):
    def setUp(self):
        # Initialize a patient with dummy ground truth data
        self.patient = Patient(dm_ground_truth={})

        # Add sessions for protocol ID 1
        self.sessions_protocol_1 = [
            Session(session_id=1, protocol_id=1, dms=[0.1, 0.2], timestamp=1000),
            Session(session_id=2, protocol_id=1, dms=[0.2, 0.3], timestamp=2000),
            Session(session_id=3, protocol_id=1, dms=[0.3, 0.4], timestamp=3000)
        ]
        for session in self.sessions_protocol_1:
            self.patient.update(session)

        # Add sessions for protocol ID 2
        self.sessions_protocol_2 = [
            Session(session_id=1, protocol_id=2, dms=[0.5, 0.6], timestamp=4000),
            Session(session_id=2, protocol_id=2, dms=[0.6, 0.7], timestamp=5000)
        ]
        for session in self.sessions_protocol_2:
            self.patient.update(session)

    def test_get_last_n_sessions(self):
        # Test for protocol ID 1
        last_2_sessions = self.patient.get_last_n_sessions(protocol_id=1, n=2)
        self.assertEqual(len(last_2_sessions), 2)
        self.assertEqual(last_2_sessions[0].timestamp, 2000)
        self.assertEqual(last_2_sessions[1].timestamp, 3000)

        # Test for protocol ID 2
        last_session = self.patient.get_last_n_sessions(protocol_id=2, n=1)
        self.assertEqual(len(last_session), 1)
        self.assertEqual(last_session[0].timestamp, 5000)

    def test_no_sessions_for_protocol(self):
        # Test for a protocol ID with no sessions
        last_sessions = self.patient.get_last_n_sessions(protocol_id=3, n=2)
        self.assertEqual(len(last_sessions), 0)

    def test_fewer_sessions_than_requested(self):
        # Test when there are fewer sessions than requested
        last_3_sessions = self.patient.get_last_n_sessions(protocol_id=2, n=3)
        self.assertEqual(len(last_3_sessions), 2)  # Only 2 sessions exist
        self.assertEqual(last_3_sessions[0].timestamp, 4000)
        self.assertEqual(last_3_sessions[1].timestamp, 5000)

if __name__ == "__main__":
    unittest.main()
