class TestHouses:

    def __init__(self, file):
        ##
    
    def get_valid_input(prompt):
        while True:
            try:
                value = int(input(prompt))
            except ValueError:
                print("Sorry, I need an integer.")
                continue
            if value < 0 or value > 10:
                print("Sorry, answer must be an integer between 0 and 10.")
                continue
            else:
                break
        return value