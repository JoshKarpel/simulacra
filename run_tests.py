import os
import argparse

import pytest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run Simulacra's tests.")
    parser.add_argument('--coverage', '-c',
                        action = 'store_true')

    args = parser.parse_args()

    cov = None
    if args.coverage:
        import coverage

        cov = coverage.coverage(branch = True, include = 'src/simulacra/*')
        cov.start()

    pytest.main([])

    if cov is not None:
        cov.stop()
        cov.save()
        print('Coverage Summary:')
        cov.report()
        report_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'cov')
        cov.html_report(directory = report_dir)
        print(f'HTML report at {os.path.join(report_dir, "index.html")}')
        cov.erase()
