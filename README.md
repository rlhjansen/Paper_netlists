

## dingen maken


## doen ##
- meer comments
- proporties netlists vergelijkbaar met verhoudingen [hier](http://heuristieken.nl/wiki/index.php?title=Chips_%26_Circuits)


## zoektermen ##
 - VLSI optimization
 - Integrated circuit

## Papers/naslagwerk ##
 - synthesis and optimization of digital circuits
 - a survey of optimization

## visualiseren
 - circuit is nu printbaar binnen terminal maar mooier = beter.
 - later verder zien

## uitzoeken ##
 - Datastructuren code style:
linter style modus aggressive (eerst kopie)
 -  pycodestyle


## conferenties/gaafheid
 - https://meta2018.sciencesconf.org/
 - http://www.airoconference.it/ods2018/topics



## todo 180507
- Update HC in case of not fully connected; always use total distance as comparison parameter.
- Visualize behavior of PPA algorithm -> mean and max value for every population.
- Distance between max and mean gives information about population. (Same goes for distribution (standard deviation, etc))
Big difference between max and mean indicates lower average fitness. If the distance gets smaller your average fitness has probably increased and can be used to support your expectations about fitness of offspring based on fitness of parents.
If difference between max and mean stays big or increases it shows that the fitness of parents is not a good predictor for the fitness of offspring.


- Create plot with HC, PPA and random sampling (or SA) progression -> take number of result evaluations as iteration
    - result evaluation; every time distance + n connections is checked for a solution
    - You can run HC first till it flatlines or takes too long and use iterations as indicator for PPA runtime or first run PPA and try to indicate the number of generations needed. Estimating the number of iterations/generations needed by plotting algorithm progress gives the best indication. Used algorithms are not deterministic so make sure you run it a few times. Since PPA is more important make sure you don't cut it of too fast.
    https://www.researchgate.net/post/What_is_the_maximum_number_of_generation_size_iteration_in_meta-heuristic_algorithms_such_as_GA
        - > fixed number of iterations for HC is not mandatory, stopping HC if no improvement is found within a certain number of iterations is also allowed, this is more complicated for PPA so just do some test runs.
    - To make sure you're in time for Essex don't do this for all generated netlists but pick a few easy ones or run all and pick the ones with the best results. This is about presenting your first findings so you don't need an extensive number of netlists to present your first assumptions.

