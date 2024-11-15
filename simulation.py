from dataclasses import dataclass

class Patient:
    """
    - initial fugl meyer
    - DM history for all the games {"protocol_name": list(list(float))}
    - Protocol history
    - TODO: come up with a protocol agnostic profile
    - DM_ground_truth: {"protocol_name": list(float)}
    """
    def __init__(self):
        pass 

    def update(self, protocol):
        # calculate the new DMs (based on the error between last dm and ground_truth, maybe add noise)

        # update dms 

        # update protocol history

        pass 


@dataclass
class Protocol:
    """
    - protocol matrix (maybe dynamic in the future)
    - DMs
    - AR (yes/no)
    """

class RecSys:
    def __init__(self):
        pass

    def recommend(patient):
        """returns list of 5 protocols
        """
        #look at last dm change for each protocol (put high for unplayed protocols)

        # turn it into a probability dist

        # randomly sample 5

        pass


#TODO: 

