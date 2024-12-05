"""
simulation.py

This module defines the `Simulator` class, which provides functionality to simulate
sessions for patients across various rehabilitation protocols. It integrates with
the `RecSys` module to leverage protocol recommendations and tracks the progression
of difficulty modulators (DMs) and other session metrics.

Classes:
    Simulator:
        Handles the simulation of patient sessions, including the generation of
        session metrics, timing attributes, and self-reported scores. Supports
        both recommended and random protocol interactions.

Functions in `Simulator`:
    - simulate_session(patient, protocol):
        Simulates a single session for a given patient and protocol.
    - model_dms(patient, protocol, adjustment_factor=0.1, noise_range=0.1):
        Models the progression of difficulty modulators (DMs) during a session.
    - model_performance():
        Generates a simulated performance time series for a session.
    - model_duration():
        Simulates the duration of a session in seconds.
    - model_self_report():
        Generates simulated self-reported scores (e.g., effort, fatigue, satisfaction).
    - use_recommendations(patient, n=5):
        Simulates the patient's interaction with the top `n` recommended protocols.
    - use_random(patient, n=5):
        Simulates the patient's interaction with `n` randomly selected protocols.

Logging:
    - Configured to log informational messages for simulated sessions and warnings
      when no ground truth DMs are found for a protocol.

Dependencies:
    - recsys.recommender: Contains the `RecSys`, `Patient`, `Protocol`, `Session`, `Metrics`,
      and `Timing` classes.
    - Python Standard Libraries: random, time, logging
    - Third-party Libraries: numpy

Example:
    # Initialize RecSys and Simulator
    recsys = RecSys(protocols=protocols)
    simulator = Simulator(recsys=recsys)

    # Simulate sessions for a patient
    patient = Patient(dm_ground_truth={
        "protocol_1": [0.5, 0.7, 0.9],
        "protocol_2": [0.6, 0.8]
    })
    simulator.use_recommendations(patient, n=3)
"""

from typing import Dict, List
import random
import time
import logging
import numpy as np
from recsys.recommender import RecSys, Patient, Protocol, Session, Metrics, Timing

# Configure logging
logging.basicConfig(level=logging.INFO)


# -------------------- Session Simulation  --------------------
class Simulator:
    """Class to simulate patient session"""

    def __init__(self, recsys: RecSys):
        self.recsys = recsys

    def run_session(self, patient: Patient, protocol: Protocol):
        """
        Simulates the progression of session DM values towards the dm_ground_truth for a given protocol.
        """
        protocol_id = protocol.id

        # ----- User behavior modelling -----
        # Generate data based on user model
        metrics = Metrics(
            dms=self.model_dms(patient, protocol),  # Generate session new DMs
            performance=self.model_performance(),  # Generate session performance
            self_report=self.model_self_report(),  # Generate self_reports
        )

        session_start = time.time()
        time_attributes = Timing(
            start=session_start,  # Session start
            end=session_start + self.model_duration(),  # Session end
            prescribed_duration=5,  # Prescribed duration
        )

        # ---- Update Patient Session ----
        # Create a new Session with the updated DM values
        new_session = Session(
            session_id=str(len(patient.get_sessions(protocol_id)) + 1),
            protocol_id=protocol_id,
            game_mode=protocol.game_modes[0],
            metric=metrics,
            timing=time_attributes,
            prescribed=True,  # Default value for now
        )

        # Update patient history
        patient.add_session(new_session)

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
        last_session = (
            patient.get_sessions(protocol_id)[-1] if patient.get_sessions(protocol_id) else None
        )
        recent_dms = (
            last_session.metric.last_dms
            if last_session
            else np.random.uniform(0.01, 0.02, size=len(target_dms)).tolist()
        )

        # Compute new DM values
        # f(n) = target - (target - current_0) * (1 - a)^n
        delta = np.array(target_dms) - np.array(recent_dms)
        noise = np.random.uniform(-noise_range, noise_range, size=len(target_dms))
        session_dms = (np.array(recent_dms) + (adjustment_factor * delta) + noise).tolist()

        return [session_dms[:] for _ in range(5)]

    def model_performance(self) -> List[float]:
        """
        We assume performance correlation with DMs.
        Returns:
            performance_timeseries (List[float]):
        """
        return [random.random() for _ in range(10)]

    def model_duration(self) -> float:
        """
        For RCT remain to see if this is therapist defined.
        Returns in session duration in seconds
        """
        minutes = random.uniform(0, 5)
        return minutes * 60

    def model_self_report(self) -> Dict[str, float]:
        """
        Returns self-reported scores.
        Three self-reports are prompted to the patient, and their scores are simulated.

        Returns:
            Dict[str, float]: A dictionary where keys represent self-report categories
            and values represent the scores (e.g., 0.0 to 10.0).
        """
        # Simulate self-report scores for three categories
        self_report_scores = {
            "effort": round(random.uniform(0.0, 10.0), 2),  # Effort self-report (0-1)
            "fatigue": round(random.uniform(0.0, 10.0), 2),  # Fatigue self-report (0-1)
            "satisfaction": round(random.uniform(0.0, 10.0), 2),  # Satisfaction self-report (0-1)
        }
        return self_report_scores

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
            self.run_session(patient, protocol)

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
            self.run_session(patient, protocol)
