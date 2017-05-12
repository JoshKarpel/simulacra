import datetime as dt

from simulacra.math import *

if __name__ == '__main__':
    low_limit = 100

    t_start = dt.datetime.now()
    print('Primes below {} via sieve:'.format(low_limit))
    for p in prime_sieve(low_limit):
        print(p, end = ', ')
    print('\nElapsed time: {}'.format(dt.datetime.now() - t_start))

    print()

    t_start = dt.datetime.now()
    print('Primes below {} via generator:'.format(low_limit))
    for p in prime_generator():
        if p > low_limit:
            break
        print(p, end = ', ')
    print('\nElapsed time: {}'.format(dt.datetime.now() - t_start))

    print()

    limit = 1000000

    t_start = dt.datetime.now()
    num_primes = 0
    for p in prime_sieve(limit):
        num_primes += 1
    print('There are {} primes less than {} (found by sieving). Elapsed time: {}'.format(num_primes, limit, dt.datetime.now() - t_start))

    t_start = dt.datetime.now()
    num_primes = 0
    for p in prime_generator():
        if p > limit:
            break
        num_primes += 1
    print('There are {} primes less than {} (found by generating). Elapsed time: {}'.format(num_primes, limit, dt.datetime.now() - t_start))

