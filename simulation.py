"""
simulation.py

This module define patient, protocol and session models and simulates sessions for different protocols in the application.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import time
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)


# ------------------ Protocol Representation ------------------
@dataclass
class Protocol:
    """
    Dataclass to store protocol information

    - protocol matrix (maybe dynamic in the future)
    - DMs
    - AR (yes/no)

    ** Maybe averge statistics for user (usage_count, avg_duration)
    """

    id: str
    name: Optional[str]
    dms: List[str]
    ar: Optional[bool]


@dataclass
class Session:
    """
    Represents a specific session of a protocol for a patient.
    """

    session_id: str
    protocol_id: str
    dms: List[float]  # Session-specific DMs
    performance: List[float]  # Session-specific DMs
    timestamp_start: Optional[int] = time.time()  # Track in UNIX time session start
    timestamp_end: Optional[int] = time.time()  # Track in UNIX time session end
    prescribed_duration: Optional[int]
    prescribed: Optional[bool] = False


# ------------------ Patient Representation ------------------
class Patient:
    """
    Class that tracks patient-protocol history

    - initial fugl meyer
    - DM history for all the games {"protocol_name": list(list(float))}
    - TODO: come up with a protocol agnostic profile
    - DM_ground_truth: {"protocol_name": list(float)}
    """

    def __init__(self, dm_ground_truth: Dict[str, List[float]]):
        self.sessions = {}  # Stores {"protocol_id": List[Session]}
        self.dm_ground_truth = dm_ground_truth  # Stores target DM values for simulation
        self.recommendation_history = []  # Stores Recommendation History

    def update(self, session: Session):
        """Update patient sessions history"""
        protocol_id = session.protocol_id

        # Update DM history
        if protocol_id not in self.sessions:
            self.sessions[protocol_id] = []

        # Add to sessions
        self.sessions[protocol_id].append(session)

    def get_last_n_sessions(self, protocol: Protocol, n: int = 2) -> List[Session]:
        """
        Retrieves the last `n` sessions for the given protocol.

        Args:
            protocol (Protocol): Protocol.
            n (int): Number of sessions to retrieve.

        Returns:
            List[Session]: Last `n` sessions for the protocol.
        """
        protocol_id = protocol.id
        if protocol_id not in self.sessions:
            return []
        return sorted(self.sessions[protocol_id], key=lambda s: s.timestamp_end)[-n:]

    def get_protocol_stats(self):
        """
        Computes statistics for a given patient

        Returns:
            List[Session]: List of performed sessions for a given protocol
        """
        metrics = {}
        for protocol_id, sessions in self.sessions.items():
            if not sessions:
                metrics[protocol_id] = {
                    "mean_performance": None,
                    "mean_difficulty": None,
                    "mean_adherence": None,
                    "session_count": 0,
                }
                continue

            df = pd.DataFrame(
                sessions
            )  # Assumes sessions is a list of dictionaries or dataclass instances
            df["duration"] = (df["timestamp_end"] - df["timestamp_start"]) / 60
            df["adherence"] = df["duration"] / df["prescribed_duration"]

            # Compute metrics
            metrics[protocol_id] = {
                "mean_performance": df["performance"].mean(),
                "mean_difficulty": df["dms"].mean(),
                "mean_adherence": df["adherence"].mean(),
                "session_count": len(sessions),
            }
        return metrics

    def save_recommendation(self, **kwargs):
        """
        Records the recommendation details.

        Args:
            **kwargs: Any keyword arguments related to recommendation details.
                Examples include:
                - delta_dm (Dict[str, float]): Delta DM values.
                - probabilities (Dict[str, float]): Probability distribution over protocols.
                - recommended_ids (List[str]): List of recommended protocol IDs.
                - extra (Dict[str, any]): Additional parameters used for the recommendation.
        """
        self.recommendation_history.append(kwargs)


# ------------------ Recommendation System ------------------
class RecSys:
    """Recommender System Class"""

    def __init__(self, protocols: List[Protocol]):
        self.protocols = protocols  # List of all protocols

    def recommend(
        self, patient: Patient, n: int = 5, params: Optional[Dict[str, any]] = None
    ) -> List[Protocol]:
        """
        Recommends `n` protocols based on computed delta DM values.

        Args:
            patient (Patient): The patient for whom protocols are recommended.
            n (int): Number of protocols to recommend.
            params (Optional[Dict[str, any]]): Additional parameters used for the recommendation.

        Returns:
            List[Protocol]: List of recommended protocols.
        """

        # -------- Check --------
        # Adjust `n` to ensure it does not exceed the available protocols
        num_protocols = len(self.protocols)
        if n > num_protocols:
            # Warning
            logging.warning(
                "Requested %d recommendations, but only %d protocols are available.",
                n,
                num_protocols,
            )
            # Update number of protocols
            n = num_protocols

        # ------- Scoring -------

        # Last delta DM values per protocol
        delta_dm = {
            protocol.id: self.compute_delta_dm(patient.get_last_n_sessions(protocol, n=2))[-1]
            for protocol in self.protocols
        }

        # Normalize delta DM values to create a probability distribution
        probabilities = dict(zip(delta_dm.keys(), self.compute_prob_dist(delta_dm.values())))

        # ------ Sampling -------
        # Randomly sample `n` protocols based on probabilities
        recommended_ids = np.random.choice(
            list(probabilities.keys()),
            size=n,
            replace=False,
            p=list(probabilities.values()),
        )

        # -------- Log ----------
        # Ensure params is a dictionary and include additional parameters
        params = {**(params or {}), "timestamp": time.time()}
        # Save the recommendation with the updated parameters
        patient.save_recommendation(
            delta_dm=delta_dm,
            probabilities=probabilities,
            recommended=recommended_ids,
            params=params,
        )

        # Retrieve and return the recommended Protocol objects
        return [protocol for protocol in self.protocols if protocol.id in recommended_ids]

    def random_sampling(
        self, patient: Patient, n: int = 5, params: Optional[Dict[str, any]] = None
    ) -> List[Protocol]:
        """
        Random sampling of protocols

        Args:
            patient (Patient): The patient for whom protocols are recommended.
            n (int): Number of protocols to sample.
            params (Optional[Dict[str, any]]): Additional parameters used for the recommendation.

        Returns:
            List[Protocol]: List of sampled protocols.
        """
        # Last delta DM values per protocol
        delta_dm = {
            protocol.id: self.compute_delta_dm(patient.get_last_n_sessions(protocol, n=2))[-1]
            for protocol in self.protocols
        }

        # ------ Random Sampling -------
        id_values = [protocol.id for protocol in self.protocols]
        sampled_ids = random.sample(id_values, n)

        # Ensure params is a dictionary and include additional parameters
        params = {**(params or {}), "timestamp": time.time()}
        # Save the recommendation with the updated parameters
        patient.save_recommendation(
            delta_dm=delta_dm,
            recommended=sampled_ids,
            params=params,
        )

        # Retrieve and return the recommended Protocol objects
        return [protocol for protocol in self.protocols if protocol.id in sampled_ids]

    @staticmethod
    def compute_prob_dist(values: List[float], epsilon: float = 1e-8) -> List[float]:
        """
        Transforms delta dms into a positive range and normalizes to sum to 1,
        ensuring no zero probabilities by adding a small constant epsilon.

        Args:
            values (List[float]): A list of last protocol delta dm for a patient.
            epsilon (float): A small constant to avoid zero probabilities.

        Returns:
            probabilities (List[float]): Sampling weights for each protocol.
        """
        # Shift to make all values positive
        min_val = min(values)
        shifted_values = [v - min_val for v in values]

        # Add epsilon to all shifted values
        adjusted_values = [v + epsilon for v in shifted_values]

        # Normalize to sum to 1
        total_value = sum(adjusted_values)
        probabilities = [v / total_value for v in adjusted_values]

        return probabilities

    @staticmethod
    def compute_delta_dm(sessions: List[Session]) -> List[float]:
        """
        Computes the total DM change based on the last n sessions.

        Args:
            sessions (List[Session]): A list of n sessions for a protocol.

        Returns:
            delta_dms_timeseries (List[float]): A list of n-1 computed average delta DM value between consecutive sessions.
        """
        if len(sessions) < 2:
            return [random.uniform(0.85, 1)]

        # Extract sessions dms
        dms_matrix = np.array([session.dms for session in sessions])
        # Compute the element-wise difference between consecutive sessions
        delta_dms = np.diff(dms_matrix, axis=0)
        # Mean delta for each session
        delta_dms_timeseries = np.mean(delta_dms, axis=1)

        return delta_dms_timeseries


# ------------------ Session Simulation  ------------------
class Simulator:
    """Class to simulate patient session"""

    def __init__(self, recsys: RecSys):
        self.recsys = recsys

    def simulate_session(self, patient: Patient, protocol: Protocol):
        """
        Simulates the progression of session DM values towards the dm_ground_truth for a given protocol.
        """
        protocol_id = protocol.id
        session_start = time.time()

        # ----- User behavior modelling -----
        # Generate session new DMs
        session_dms = self.model_dms(patient, protocol)
        # Generate session performance
        session_performance = self.model_performance(patient, protocol)
        # Generate session duration
        duration = self.model_duration(patient, protocol)

        # ---- Update Patient Session ----
        # Create a new Session with the updated DM values
        new_session = Session(
            session_id=str(len(patient.sessions.get(protocol_id, [])) + 1),
            protocol_id=protocol_id,
            dms=session_dms,
            performance=session_performance,
            timestamp_start=session_start,
            timestamp_end=session_start + duration,
            prescribed=True,  # Default value for now
        )

        # Update patient history
        patient.update(new_session)

    def model_dms(
        self,
        patient: Patient,
        protocol: Protocol,
        adjustment_factor: float = 0.1,
        noise_range: float = 0.1,
    ) -> List[float]:
        """
        Patient expected session DMs model

        Args:
            patient (Patient): Patient performing the session.
            protocol (Protocol): Protocol to be used.
            adjustment_factor (float): Learning factor.
            noise_range (float): Noise applied to the dms model

        Returns:
            session_dms (List[float]): DMs after session
        """
        protocol_id = protocol.id

        # Check for target DMs
        target_dms = patient.dm_ground_truth.get(protocol_id)
        if not target_dms:
            logging.warning("No ground truth found for protocol %s", protocol_id)
            return []

        # Get last session DMs
        last_session = patient.get_last_n_sessions(protocol, 1)
        last_dms = (
            last_session[0].dms
            if last_session
            else np.random.uniform(0.01, 0.02, size=len(target_dms)).tolist()
        )

        # Compute new DM values
        # f(n) = target - (target - current_0) * (1 - a)^n
        delta = np.array(target_dms) - np.array(last_dms)
        noise = np.random.uniform(-noise_range, noise_range, size=len(target_dms))
        session_dms = (np.array(last_dms) + (adjustment_factor * delta) + noise).tolist()

        return session_dms

    def model_performance(self, patient: Patient, protocol: Protocol):
        """ """
        raise NotImplementedError("Implement DMs model!")

    def model_duration(self, patient: Patient, protocol: Protocol):
        """ """
        raise NotImplementedError("Implement DMs model!")

    def use_recommendations(self, patient: Patient, n: int = 5):
        """
        Simulates the patient's interaction with the recommended protocols.

        Args:
            patient (Patient): The patient who will execute the recommendations.
            n (int): Number of protocols to recommend and execute.
            adjustment_factor (float): How much the DMs adjust toward their target.
            noise_range (float): The range of randomness in DM adjustments.
        """
        # Get recommended protocols
        recommended_protocols = self.recsys.recommend(patient, n=n)

        # Simulate sessions for each recommended protocol
        for protocol in recommended_protocols:
            logging.info("Simulating session for protocol %s: %s", protocol.id, protocol.name)
            self.simulate_session(patient, protocol)

    def use_random(self, patient: Patient, n: int = 5):
        """
        Simulates the patient's interaction with the recommended protocols.

        Args:
            patient (Patient): The patient who will execute the recommendations.
            n (int): Number of protocols to recommend and execute.
            adjustment_factor (float): How much the DMs adjust toward their target.
            noise_range (float): The range of randomness in DM adjustments.
        """
        # Get recommended protocols
        recommended_protocols = self.recsys.random_sampling(patient, n=n)

        # Simulate sessions for each recommended protocol
        for protocol in recommended_protocols:
            logging.info("Simulating session for protocol %s: %s", protocol.id, protocol.name)
            self.simulate_session(patient, protocol)
