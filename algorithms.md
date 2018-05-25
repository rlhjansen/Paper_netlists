## notes on Plant propagation ##

 - Basis for this research
    - The Seminal Paper for this project: https://www.researchgate.net/publication/252321319_Nature-Inspired_Optimisation_Approaches_and_the_New_Plant_Propagation_Algorithm?enrichId=rgreq-33e633440053db9615312bdfbebe3b49-XXX&enrichSource=Y292ZXJQYWdlOzI1MjMyMTMxOTtBUzo1MjI4MjY2MDIyMjE1NjhAMTUwMTY2Mjk4MzYzNw%3D%3D&el=1_x_3&_esc=publicationCoverPdf
    - Further work:
        1. [A novel plant propagation algorithm](https://www.researchgate.net/publication/293822668_A_Novel_Plant_Propagation_Algorithm_Modifications_and_Implementation)
        2. [Plant propagation inspired algorithms](https://www.researchgate.net/publication/316065481_Plant_propagation-inspired_algorithms?_iepl%5BgeneralViewId%5D=xSVTZjnDNmlUBGn0h8ddm2ctnJ9c3OniFZjY&_iepl%5Bcontexts%5D%5B0%5D=searchReact&_iepl%5BviewId%5D=xV4c9tIVLjQhJqEkv6HoE71qEL6C8msnF04z&_iepl%5BsearchType%5D=publication&_iepl%5Bdata%5D%5BcountLessEqual20%5D=1&_iepl%5Bdata%5D%5BinteractedWithPosition1%5D=1&_iepl%5Bdata%5D%5BwithoutEnrichment%5D=1&_iepl%5Bposition%5D=1&_iepl%5BrgKey%5D=PB%3A316065481&_iepl%5BtargetEntityId%5D=PB%3A316065481&_iepl%5BinteractionType%5D=publicationTitle)

 - Strawberry metaphor
    - Strawberries explore & exploit their environment.
    - This is akin to optimization
    - Strawberry plants make shoots to propagate themselves
    - The length & amount of shoots, from now on runners, is determined by the fitness i the current location.
    - Good locations in the environment cause the strawberry to make more and closer runners.
    - 'Bad' locations in the environment cause the strawberry to produce fewer but longer runners.

- algorithms characteristics
    - Each sample solution(it's parameters) is moddeled as a plant
    - It has a certain fitness based on function evaluation
    - In determining how much/long shoots to send a comparison is made to other currently living plants
    - The best p plants propagate themselves
    - The best e plants stay alive

- algorithm procedure:
    - Create initial (new) population and empty set for last generation
    - Loop till max generation
        1. Evaluate new population
        2. Select propagators from current & last generation
        3. Create new population.
        4. Select not dieing plants from the current population and assign them to last generation
    - Return input with best function evaluation


- personal notes

    The original paper makes use of (relative) fitness function in regard to other plants in the generation. This however is not entirely the natural situation. this would be akin to measuring only how much sunlight a plant get, being overshadowed by other environment factors (trees, standard fitness) as fitness and other strawberry plants (relative fitness). but does not take into accoun the fertility of the ground, how much minerals are (left) to gather, causing the strawberry plants to over cluster if the solutionspace is steep, since worse off plants will be indirectly killed by better plants through suboptimal function evaluation when moving away from the 'good' patch.

    In light of this a future research point will be to map the results onto a 'soil' as well, determining life or death not only by relative function evaluation as determined by other strawberries, but also by relative location to other plants, preferring less occupied ground (which should therefore be more fertile).

    Think of this as strawberries planted in plant borders of a garden. When the ground is overly saturated by strawberries apparent will be made to cross gravel paths between plant borders in the garden. some strawberry plants even are able to survive in the gravel paths trough nutrition from their motherplant and water from rainfall. numerically speaking, it performs worse in fitness than other plants, but through its survival other unknown areas can become new territory to explore. in the current implementation of the algorithm, such a propagation strategy is impossible. One could argue that this is a problem that could be solved by tuning runner & distance parameters. if this aspect is uncertain (what is distance in ordering, especially when orders are able to produce exact same results)

## notes on Artificial Bee Colony optimization ##
 - in progress

## future research notes ##
 - looking how mushrooms & other plants with root networks spread.

### link to markup ###
[github page for markup cheatsheet](https://github.com/tchapi/markdown-cheatsheet/blob/master/README.md)
[other link to cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)