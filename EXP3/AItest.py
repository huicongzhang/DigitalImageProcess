import random
def trial():
        funds = 10
        plays = 0
        while funds >= 1:
                funds -=1
                plays += 1
                slots = [random.choice(["bar", "bell", "lemon", "cherry"]) for i in range(3)]
                if slots[0] == slots[1]:
                        if slots[1] == slots[2]:
                            num_equal = 3
                        else:
                            num_equal = 2
                else:
                    num_equal = 1
                if slots[0] == "cherry":
                    funds += num_equal 
                elif num_equal == 3:
                    if slots[0] == "bar":
                        funds += 20
                    elif slots[0] == "bell":
                        funds += 15
                    else:
                        funds += 5
        return plays
def test(trials):
    results = [trial() for i in range(trials)]
    mean = sum(results) / float(trials)
    median = sorted(results)[int(trials/2)]
    print ("{} trials: mean={}, median={}".format(trials, mean, median))
test(1000)
