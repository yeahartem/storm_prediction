from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection
import tempfile
import os, subprocess
import json
import sys

def download_cmips(
    path_to_context='context.json', 
    path_to_download_dir='data'
):
    if os.path.exists(path_to_download_dir):
        if os.path.isdir(path_to_download_dir):
            pass
        else:
            os.mkdir(path_to_download_dir)
    else:
        os.mkdir(path_to_download_dir)

    lm = LogonManager()
    lm.logoff()
    lm.is_logged_on()
    myproxy_host = 'esgf-node.llnl.gov'
    lm.logon(username='isartemm', password='password', hostname=myproxy_host, bootstrap=True)
    lm.is_logged_on()

    conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=False)

    with open(path_to_context) as fs:
        context = json.load(fs)

    ctx = conn.new_context(project=context['project'], 
                        model=context['model'], 
                        experiment=context['experiment'],
                        cf_standard_name="wind_speed",   
                        time_frequency=context['time_frequency'],
                        realm=context['realm'],
                        ensemble=context['ensemble'],
                        version=context['version'])
    ds = ctx.search()[0]

    fc = ds.file_context()
    wget_script_content = fc.get_download_script()
    fd, script_path = tempfile.mkstemp(suffix='.sh', prefix='download-')
    with open(script_path, "w") as writer:
        writer.write(wget_script_content)
    os.close(fd)

    os.chmod(script_path, 0o750)
    subprocess.check_output("{}".format(script_path), cwd=path_to_download_dir)

    ctx = conn.new_context(project=context['project'], 
                        model=context['model'],
                        experiment=context['experiment'],   
                        time_frequency=context['time_frequency'],
                        realm=context['realm'],
                        ensemble=context['ensemble'],
                        version=context['version'])
    ds = ctx.search()[0]

    fc = ds.file_context()
    wget_script_content = fc.get_download_script()
    fd, script_path = tempfile.mkstemp(suffix='.sh', prefix='download-')
    with open(script_path, "w") as writer:
        writer.write(wget_script_content)
    os.close(fd)

    os.chmod(script_path, 0o750)
    subprocess.check_output("{}".format(script_path), cwd=path_to_download_dir)

    

if __name__ == '__main__':
    if len(sys.argv) == 3:
        download_cmips(path_to_context=sys.argv[1], 
                        path_to_download_dir = sys.argv[2])
    else:
        download_cmips()