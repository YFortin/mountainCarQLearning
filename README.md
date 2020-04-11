# mountainCarQLearning

Q Learning implementation for openai Gym Mountain Car environment.

With default settings, should succeed at around the 1300th iteration. By the end of the 4000th iteration, should be able to win the game about 90% of the time.

If what matter is to get the first win as soon as possible, lower the `NUMBER_OF_ITERATIONS` to about a 1000 and you can expect a win at about the 500th iteration.

#### Install and run
```
pip install -r requirements.txt
python main.py
```

Strongly inspired by Sentdex https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/ 
