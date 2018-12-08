import ppasela as sela

if __name__ == '__main__':
    PS = sela.PPASELA(c, cX, n, nX, x, y, tag, iters=10000, generated=generated, pop_cut=p, arbitrary=a, distance=d, ordering="div", best_percent=bp)
    PS.run_algorithm()
