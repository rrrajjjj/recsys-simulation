"""
recommender.py

This module defines classes and methods for model patient sessions,
managing protocols, and implementing a recommendation system for protocols
in a rehabilitation application. It includes representations for patient
profiles, session data, and protocol metadata, as well as functionality
to recommend or sample protocols based on computed metrics.

Classes:
    Protocol:
        Represents the configuration and metadata of a rehabilitation protocol,
        including difficulty modulators, game modes, and augmented reality features.

    Timing:
        Encapsulates session timing information, including start and end times,
        prescribed duration, and methods to compute duration and adherence.

    Metrics:
        Tracks session-specific metrics such as difficulty modulators (DMs),
        performance scores, and patient self-reports.

    Session:
        Represents a single protocol session for a patient, combining timing,
        metrics, and session-specific attributes.

    Patient:
        Maintains the history of a patient's sessions, their protocol-related
        data, and recommendation history. Provides methods for updating sessions
        and computing statistics.

    RecSys:
        Implements the recommendation system, providing methods to recommend
        protocols based on patient session data and delta DM values. Includes
        utilities for scoring protocols, random sampling, and probability computation.

Logging:
    Configured to log informational messages, including warnings when the
    requested number of recommendations exceeds available protocols.

Dependencies:
    - Python Standard Libraries: random, time, logging
    - Third-party Libraries: numpy, scipy
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
import time
import logging
import numpy as np
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(level=logging.INFO)


# ------------------ Protocol Representation ------------------
@dataclass
class Protocol:
    """
    Represents protocol information and configuration.

    The `Protocol` class stores metadata and attributes related to a specific protocol
    used in sessions. It includes details about the protocol's features, difficulty
    modulators (DMs), game modes, and augmented reality (AR). It also may serve
    as a placeholder for computed aggregate statistics, such as average adherence
    and saturation index, which can be derived from user session data.

    Attributes:
        id (str): Unique identifier for the protocol.
        name (Optional[str]): Human-readable name of the protocol.
        features (Dict[str, float]): A dictionary of features describing the protocol
            (e.g., "memory": 0.8, "attention": 0.6).
        game_mode (List[str]): A list of game modes associated with the protocol.
        dms (List[str]): A list of names or identifiers for difficulty modulators
            used in the protocol to adjust its difficulty dynamically.
        ar (Optional[bool]): Indicates whether protocol is augmented reality (AR)

    Notes:
        - This class is designed to represent the static properties of a protocol.
        - Dynamic statistics such as average adherence or saturation index can be
          computed and added as attributes in the future.

    Example:
        protocol = Protocol(
            id="protocol_123",
            name="Rehabilitation Protocol A",
            features={"memory": 0.8, "attention": 0.6},
            game_mode=["single_player", "cooperative"],
            dms=["dm1", "dm2"],
            ar=True
        )
    """

    id: str
    name: Optional[str]
    profile: Dict[str, float]
    game_modes: List[str]
    dms: List[str]
    ar: Optional[bool]


# ------------------ Session Representation ------------------
@dataclass
class Timing:
    """
    Represents time related session attributes

    The `Timing` class encapsulates information about the temporal aspects of a session,
    including start time, end time, and the prescribed duration. It provides methods to
    compute the actual session duration and the adherence to the prescribed duration.

    Attributes:
        start (int): The session's start time as a UNIX timestamp. Defaults to the current time.
        end (int): The session's end time as a UNIX timestamp. Defaults to 0, indicating
            the session has not ended.
        prescribed_duration (int): The prescribed duration of the session in seconds.

    Methods:
        duration() -> int:
            Computes the session duration in minutes.
        adherence() -> float:
            Computes the adherence to the prescribed duration.

    Example:
        timing = Timing(start=1638450000, end=1638450300, prescribed_duration=300)

        # Compute session duration
        print(timing.duration())  # Outputs: 300 (seconds)

        # Compute adherence
        print(timing.adherence())  # Outputs: 1.0 (100% adherence)
    """

    start: int = field(default_factory=lambda: int(time.time()))
    end: int = 0
    prescribed_duration: int = 0

    @property
    def date(self) -> str:
        """
        Converts the start time to a human-readable date format (YYYY/MM/DD).
        """
        return datetime.fromtimestamp(self.start).strftime("%Y/%m/%d")

    @property
    def duration(self) -> int:
        """
        Computes the duration in minutes between the start and end times.
        If the end time is not set, calculates duration up to the current time.
        """
        return (self.end - self.start) / 60

    @property
    def adherence(self) -> float:
        """
        Computes the adherence to the prescribed duration as the ratio of actual duration
        to prescribed duration. Raises a ValueError if `prescribed_duration` is not greater
        than 0.
        """
        if self.prescribed_duration <= 0:
            raise ValueError("Prescribed duration must be greater than 0 to calculate adherence.")

        actual_duration = self.duration
        return actual_duration / self.prescribed_duration


@dataclass
class Metrics:
    """
    Represents session data and subjective metrics associtaed with the session.

    This class tracks key data points from a session, including difficulty
    modulators (DMs), objective performance scores, and self-reported feedback.

    Attributes:
        dms (List[List[float]]): A list of lists where each sublist corresponds to
            the values of a specific difficulty modulator (DM) that controls the
            protocol's difficulty. Each sublist contains the DM values recorded
            across the session.
            Example:
                - [[0.5, 0.6, 0.7], [0.8, 0.85, 0.9]] for two DMs with their
                  respective values throughout the session.
        performance (List[float]): A list of performance scores recorded at different
            stages of the session. Scores are typically normalized values
            (e.g., between 0.0 and 1.0) representing the patient's performance.
        self_report (List[float]): A list of subjective self-reported evaluations
            provided by the patient, such as effort, fatigue, or satisfaction scores.
            These scores are used for qualitative insights into the session experience.

    Example:
        metrics = Metrics(
            dms=[[0.5, 0.6, 0.7], [0.8, 0.85, 0.9]],
            performance=[0.75, 0.80, 0.85],
            self_report=[3.0, 7.0, 8.0]
        )
    """

    dms: List[List[float]] = field(default_factory=list)
    performance: List[float] = field(default_factory=list)
    self_report: List[float] = field(default_factory=list)

    @property
    def last_dms(self):
        """
        Returns the most recent DM values, or None if no DMs exist.
        """
        return self.dms[-1] if self.dms else None

    @property
    def avg_report(self):
        """
        Returns the average self-report score, or None if no scores exist.
        """
        return float(np.mean(self.self_report)) if self.self_report else None


@dataclass
class Session:
    """
    Represents a specific session of a protocol for a patient.

    A session tracks the execution details of a protocol for a given patient,
    including metrics such as difficulty modulators (DMs), performance, and
    associated self-reported scores. It also captures timing information and
    whether the session was prescribed as part of the rehabilitation plan.

    Attributes:
        session_id (str): Unique identifier for the session.
        protocol_id (str): Identifier of the associated protocol.
        metric (Metrics): An object containing metrics for the session, such as
            difficulty modulators (DMs), performance scores, and self-reports.
        timing (Timing): An object encapsulating session timing details, such as
            start time, end time, prescribed duration, and methods to compute
            duration and adherence.
        prescribed (bool): Indicates whether this session was prescribed by a
            clinician or part of a predefined plan.

    Example:
        metric = Metrics(dms=[0.5, 0.6], performance=[0.8], self_report=[0.9])
        timing = Timing(start=1638450000, end=1638453600, prescribed_duration=60)
        session = Session(
            session_id="session_001",
            protocol_id="protocol_123",
            metric=metric,
            timing=timing,
            prescribed=True
        )
    """

    session_id: str
    protocol_id: str
    game_mode: str
    metric: Metrics
    timing: Timing
    prescribed: bool


# ------------------ Patient Representation -------------------
class Patient:
    """
    Class that tracks patient-protocol history

    - initial fugl meyer
    - DM history for all the games {"protocol_name": list(list(float))}
    - TODO: come up with a protocol agnostic profile
    - DM_ground_truth: {"protocol_name": list(float)}
    """

    def __init__(self):
        self.profile: Dict[str, float] = {}
        self._sessions: Dict[(str, str), List[Session]] = {}

    def add_session(self, session: Session):
        """
        Adds a session to the internal storage, indexed by protocol ID and session date.

        Args:
            session (Session): The session object to be added. It must have `protocol_id` and `timing.date` attributes.

        Behavior:
            - If a session with the same protocol ID and date does not exist, a new entry is created.
            - If a session with the same protocol ID and date exists, the session is appended to the list of sessions for that key.
        """
        key = (session.protocol_id, session.timing.date)  # Tuple key: (protocol_id, session_date)
        if key not in self._sessions:
            self._sessions[key] = []
        self._sessions[key].append(session)

    def get_sessions(self, protocol_id: str = None, date: str = None) -> List[Session]:
        """
        Retrieves a list of sessions based on the provided protocol ID and/or date.

        Args:
            protocol_id (str, optional): The ID of the protocol for which sessions are requested.
                - If `None`, sessions for all protocols are included.
            date (str, optional): The date for which sessions are requested (in "YYYY-MM-DD" format).
                - If `None`, sessions for all dates are included.

        Returns:
            List[Session]: A list of sessions matching the specified criteria.

        Behavior:
            - If both `protocol_id` and `date` are `None`, returns all sessions.
            - If only `protocol_id` is provided, returns sessions for the given protocol across all dates.
            - If only `date` is provided, returns sessions for all protocols on the given date.
            - If both `protocol_id` and `date` are provided, returns sessions for the specified protocol and date.
            - If no sessions match the criteria, returns an empty list.

        Example Usage:
            - Retrieve all sessions:
                `sessions = get_sessions()`
            - Retrieve sessions for protocol "protocol_1":
                `sessions = get_sessions(protocol_id="protocol_1")`
            - Retrieve sessions for date "2024-01-01":
                `sessions = get_sessions(date="2024-01-01")`
            - Retrieve sessions for protocol "protocol_1" on "2024-01-01":
                `sessions = get_sessions(protocol_id="protocol_1", date="2024-01-01")`
        """
        if protocol_id is None and date is None:
            # Return all sessions
            return [session for sessions in self._sessions.values() for session in sessions]
        elif protocol_id is None:
            # Return sessions for a specific date
            return [
                session
                for (pid, d), sessions in self._sessions.items()
                if d == date
                for session in sessions
            ]
        elif date is None:
            # Return sessions for a specific protocol
            return [
                session
                for (pid, d), sessions in self._sessions.items()
                if pid == protocol_id
                for session in sessions
            ]
        else:
            # Return sessions for a specific protocol and date
            return self._sessions.get((protocol_id, date), [])

    def get_last_session_date(self) -> Optional[str]:
        """
        Retrieves the date of the last session.

        Returns:
            Optional[str]: The date of the last session in 'YYYY-MM-DD' format, or None if no sessions exist.
        """
        if not self._sessions:
            return None

        # Extract all dates directly from the dictionary keys
        all_dates = [date for _, date in self._sessions]

        # Find the most recent date
        latest_date = max(all_dates)

        # Return the latest date as a string in 'YYYY-MM-DD' format
        return latest_date

    def compute_for_protocols(
        self,
        protocols,
        logic: Callable[[Any], float],
        default_value: float = 0.0,
        min_sessions: int = 0,
    ) -> Dict[str, float]:
        """
        Generalized method to compute values for protocols based on specific logic.

        Args:
            protocols: A list of protocol objects.
            logic (Callable[[Any], float]): A function defining the computation logic for each protocol's sessions.
            default_value (float): The default value to return for protocols with no sessions or insufficient sessions.
            min_sessions (int): The minimum number of sessions required to compute the value.

        Returns:
            Dict[str, float]: A dictionary mapping protocol IDs to computed values.
        """
        return {
            protocol.id: (
                logic(sessions)
                if (sessions := self.get_sessions(protocol_id=protocol.id))
                and len(sessions) >= min_sessions
                else default_value
            )
            for protocol in protocols
        }


# ------------------ Recommendation System --------------------
class RecSys:
    """Recommender System Class"""

    def __init__(self, protocols: List[Protocol]):
        self.protocols = protocols  # List of all protocols
        self.log = {}

    def recommend(
        self, patient: Patient, n: int = 5, params: Optional[Dict[str, any]] = None
    ) -> List[Protocol]:
        """
        Recommends `n` protocols based on computed delta DM values.

        Scoring Function:

        .. math::

            Score = a \\cdot PPF + b \\cdot \\Delta dm + c \\cdot association_{subj} + d \\cdot adherence

        Where:
            - :math:`PPF` is the Patient Protocol Fit.
            - :math:`\\Delta dm` represents recent difficulty modulator changes.
            - :math:`association_{subj}` is the association with subjective assessment (e.g., mood or motivation).
            - :math:`adherence` refers to the adherence ratio (completed duration over prescribed duration).

        If the patient is tired or unmotivated, protocols with high associations with positive moods
        and relatively easy recent changes (high :math:`\\Delta dm`) may be prioritized.

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
        # Weights
        weights = {"a": 0.01, "b": 0.5, "c": 0.1}

        # Compute Patient Protocol fit
        ppf = self.compute_ppf(patient.profile)

        # Compute Patient Protocol adherence
        adherence = patient.compute_for_protocols(
            self.protocols,
            logic=lambda sessions: float(
                np.mean([session.timing.adherence for session in sessions])
            ),
            default_value=1.0,
        )

        # Compute recent average delta dm per protocol Dict[str, float]
        delta_dm = patient.compute_for_protocols(
            self.protocols,
            logic=lambda sessions: (
                float(
                    np.mean(np.diff([session.metric.last_dms for session in sessions], axis=0)[-1])
                )
            ),
            default_value=1.0,
            min_sessions=2,
        )

        # Compute protocol scoring formula
        scores = self.compute_scoring(ppf, delta_dm, adherence, weights)

        # Normalize to prob dist
        probabilities = self.compute_prob_dist(scores)

        # ------ Sampling -------
        # Randomly sample `n` protocols based on probabilities
        recommended_ids = np.random.choice(
            list(probabilities.keys()),
            size=n,
            replace=False,
            p=list(probabilities.values()),
        )

        # -------- Log ----------

        # Retrieve and return the recommended Protocol objects
        return [protocol for protocol in self.protocols if protocol.id in recommended_ids]

    def compute_ppf(self, patient_profile: Dict[str, float]) -> Dict[str, float]:
        """
        Computes the Patient Protocol Fit (PPF) as the similarity between the patient's profile
        and the profiles of all protocols in `self.protocols`.

        Args:
            patient_profile (Dict[str, float]): The patient's profile represented as a dictionary of feature values.

        Returns:
            Dict[str, float]: A dictionary mapping protocol IDs to their similarity scores.
        """
        # Convert patient profile and protocol profiles to vectors
        patient_vector = np.array(list(patient_profile.values()))
        protocol_vectors = np.array([list(p.profile.values()) for p in self.protocols])

        # Compute cosine similarity
        # Cosine similarity = 1 - cosine distance
        similarities = 1 - cdist([patient_vector], protocol_vectors, metric="cosine").flatten()

        # Map protocol IDs to similarity scores
        ppf_scores = {p.id: similarity for p, similarity in zip(self.protocols, similarities)}

        return ppf_scores

    @staticmethod
    def compute_scoring(
        ppf: Dict[str, float],
        delta_dm: Dict[str, float],
        adherence: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Computes the score for each protocol based on the given weights and input dictionaries.

        Args:
            ppf (Dict[str, float]): Dictionary of Patient Protocol Fit values.
            delta_dm (Dict[str, float]): Dictionary of delta DM values.
            association_subj (Dict[str, float]): Dictionary of subjective assessment associations.
            adherence (Dict[str, float]): Dictionary of adherence values.
            weights (Dict[str, float]): Weights for the scoring function (a, b, c, d).

        Returns:
            Dict[str, float]: A dictionary mapping protocol IDs to their computed scores.
        """
        scores = {
            protocol_id: (
                weights["a"] * ppf.get(protocol_id, 0)
                + weights["b"] * delta_dm.get(protocol_id, 0)
                + weights["c"] * adherence.get(protocol_id, 0)
            )
            for protocol_id in set(ppf.keys()).union(delta_dm.keys(), adherence.keys())
        }
        return scores

    @staticmethod
    def compute_prob_dist(values: Dict[str, float], epsilon: float = 1e-8) -> Dict[str, float]:
        """
        Transforms protocol scores into a positive range and normalizes to sum to 1,
        ensuring no zero probabilities by adding a small constant epsilon.

        Args:
            values (Dict[str, float]): A dictionary of protocol scores (keys are protocol IDs, values are scores).
            epsilon (float): A small constant to avoid zero probabilities.

        Returns:
            Dict[str, float]: Sampling weights (probabilities) for each protocol, normalized to sum to 1.
        """
        if not values:
            return {}

        # Shift to make all values positive
        min_val = min(values.values())
        shifted_values = {k: v - min_val for k, v in values.items()}

        # Add epsilon to all shifted values
        adjusted_values = {k: v + epsilon for k, v in shifted_values.items()}

        # Normalize to sum to 1
        total_value = sum(adjusted_values.values())
        probabilities = {k: v / total_value for k, v in adjusted_values.items()}

        return probabilities
