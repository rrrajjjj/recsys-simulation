from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import numpy as np
import time

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
    id: int
    name: Optional[str]
    dms: List[str]
    ar: Optional[bool]


@dataclass
class Session:
    """
    Represents a specific session of a protocol for a patient.
    """
    session_id: int
    protocol_id: int
    dms: List[float]  # Session-specific DMs
    timestamp: Optional[int] = time.time()  # Optional: Track in UNIX time when the session occurred
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
        self.sessions = {} # Stores {"protocol_id": {"session_id": [0.1, 0.2, 0.3]}}
        self.recommendation_history = [] # Stores Recommendation History
        self.dm_ground_truth = dm_ground_truth # Stores target DM values for simulation

    def update(self, session: Session):
        protocol_id = str(session.protocol_id)

        # Update DM history
        if protocol_id not in self.sessions:
            self.sessions[protocol_id] = []

        # Add to sessions
        self.sessions[protocol_id].append(session)

    def get_last_n_sessions(self, protocol_id: int, n: int = 2) -> List[Session]:
        """
        Retrieves the last `n` sessions for the given protocol.

        Args:
            protocol_id (int): ID of the protocol.
            n (int): Number of sessions to retrieve.

        Returns:
            List[Session]: Last `n` sessions for the protocol.
        """
        protocol_id = str(protocol_id)
        if protocol_id not in self.sessions:
            return []
        return sorted(self.sessions[protocol_id], key=lambda s: s.session_id)[-n:]

    def save_recommendation(self, probabilities: Dict[int, float], recommended_ids: List[str], params: Dict[str, any]):
        """
        Records the recommendation details.

        Args:
            probabilities (Dict[str, float]): Probability distribution over protocols.
            recommended_ids (List[str]): List of recommended protocol IDs.
            params (Optional[Dict[str, any]]): Additional parameters used for the recommendation.
        """
        self.recommendation_history.append({
            "prob_dist": probabilities,
            "recommended": recommended_ids,
            "params": params
        })


# ------------------ Recommendation System ------------------
class RecSys:
    def __init__(self, protocols: List[Protocol]):
        self.protocols = protocols  # List of all protocols

    @staticmethod
    def compute_delta_dm(sessions: List[Session]) -> float:
        """
        Computes the total DM change based on the last two sessions.

        Args:
            sessions (List[Session]): A list of sessions for a protocol.

        Returns:
            float: The computed delta DM value.
        """
        if not sessions:
            # High delta for unplayed protocols
            return random.uniform(0.7, 1.0)
    
        if len(sessions) == 1:
            # If there's only one session, return the sum of its DM values
            return sum(sessions[0].dms)
    
        # Compute cumulative delta across all consecutive session pairs
        total_delta = 0.0
        for i in range(1, len(sessions)):
            current_session = sessions[i]
            previous_session = sessions[i - 1]
    
            # Compute the element-wise absolute difference
            pair_delta = sum(abs(last - prev) for last, prev in zip(current_session.dms, previous_session.dms))
            total_delta += pair_delta
    
        return total_delta

    def recommend(self, patient: Patient, n: int = 5, params: Optional[Dict[str, any]] = None) -> List[Protocol]:
        """
        Recommends `n` protocols based on computed delta DM values.

        Args:
            patient (Patient): The patient for whom protocols are recommended.
            n (int): Number of protocols to recommend.
            params (Optional[Dict[str, any]]): Additional parameters used for the recommendation.

        Returns:
            List[Protocol]: List of recommended protocols.
        """
        delta_dm = {}
        
        # Compute delta DM for each protocol
        for protocol in self.protocols:
            protocol_id = protocol.id
            sessions = patient.get_last_n_sessions(protocol_id, n=2)  # Retrieve last 2 sessions
            delta_dm[protocol_id] = self.compute_delta_dm(sessions)  # Compute delta

        # Normalize delta DM values to create a probability distribution
        total_delta = sum(delta_dm.values())
        probabilities = {k: v / total_delta for k, v in delta_dm.items()}
        
        # Randomly sample `n` protocols based on probabilities
        recommended_ids = np.random.choice(
            list(probabilities.keys()),
            size=n,
            replace=False,
            p=list(probabilities.values())
        )

        # Include recommendation size in parameters
        if params is None:
            params = {}
        params['n'] = n

        # Record recommendation history, including parameters
        patient.save_recommendation(probabilities, recommended_ids, params)

        # Retrieve and return the recommended Protocol objects
        return [protocol for protocol in self.protocols if str(protocol.id) in recommended_ids]
 