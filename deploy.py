import os
import shutil


def get_dirs(directory, contents):
    """Return a tuple containing the names of the subdirectories of directory."""
    return tuple(p for p in contents if not os.path.isfile(os.path.join(directory, p)))


if __name__ == '__main__':
    shutil.rmtree('deploy', ignore_errors = True)  # clear existing deply directory to prevent any clashes

    # copy directories out of cwd, ignoring hidden folders and ignoring any subdirectories
    for d in (d for d in os.listdir(os.getcwd()) if not d.startswith('.') and d not in ('deploy', 'docs', 'dispersion') and os.path.isdir(d)):
        t = os.path.join('deploy', d)
        shutil.copytree(d, t, ignore = get_dirs)

    # go into the new 'deploy' directory and make gztar versions of each subdirectory we found
    os.chdir('deploy')
    for d in os.listdir(os.getcwd()):
        shutil.make_archive(d, format = 'gztar', base_dir = d)

    # go back up
    os.chdir('..')

    # copy various scripts into the deploy directory
    shutil.copy2('compy/scripts/make_dag.py', 'deploy')

    shutil.copy2('ionization/scripts/run_sim.py', 'deploy')
    shutil.copy2('ionization/scripts/run_sim.sh', 'deploy')
    shutil.copy2('ionization/scripts/sync_and_process.py', 'deploy')

    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__hyd__scan_width_phase_fluence.py', 'deploy')
    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__lide__scan_width_phase_fluence.py', 'deploy')
    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__vide__scan_width_phase_fluence.py', 'deploy')
    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__fsw__scan_width_phase_fluence.py', 'deploy')
    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__hyd__convergence_test.py', 'deploy')

    # custom paths for copying the deploy dir into Dropbox for sync with other computers
    for path in (r'C:\Users\Josh\Dropbox\Research\deploy',  # laptop
                 r'D:\Dropbox\Research\deploy'  # desktop
                 ):
        try:
            shutil.rmtree(path, ignore_errors = True)
            shutil.copytree('deploy', path)
        except Exception as e:
            print(e)
