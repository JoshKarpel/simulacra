import os
import shutil


def get_dirs(directory, contents):
    return tuple(p for p in contents if not os.path.isfile(os.path.join(directory, p)))


if __name__ == '__main__':
    shutil.rmtree('deploy', ignore_errors = True)

    for d in (d for d in os.listdir(os.getcwd()) if not d.startswith('.') and d not in ('deploy', 'docs') and os.path.isdir(d)):
        t = os.path.join('deploy', d)
        shutil.copytree(d, t, ignore = get_dirs)

    os.chdir('deploy')
    for d in os.listdir(os.getcwd()):
        shutil.make_archive(d, format = 'gztar', base_dir = d)

    os.chdir('..')

    shutil.copy2('ionization/scripts/run_sim.py', 'deploy')
    shutil.copy2('ionization/scripts/run_sim.sh', 'deploy')
    shutil.copy2('ionization/scripts/sync_and_process.py', 'deploy')

    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__hyd__scan_width_phase_fluence.py', 'deploy')
    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__ide__scan_width_phase_fluence.py', 'deploy')
    shutil.copy2('ionization/scripts/cluster_job_creation/create_job__fsw__scan_width_phase_fluence.py', 'deploy')

    for path in (r'C:\Users\Josh\Dropbox\Research\deploy',
                 r'D:\Dropbox\Research\deploy'):
        try:
            shutil.rmtree(path, ignore_errors = True)
            shutil.copytree('deploy', path)
        except Exception as e:
            print(e)
