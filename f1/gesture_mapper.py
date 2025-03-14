from movements import Movement

class_map = {
    0: Movement.FORWARD,
    1: Movement.LEFT,
    2: Movement.RIGHT,
    3: Movement.STOP
}

def map_class_to_action(predicted_class: int) -> Movement:
    """
    Takes the integer label from the classifier and 
    returns the corresponding Movement enum value.
    Defaults to STOP if not recognized.
    """
    return class_map.get(predicted_class, Movement.STOP)
