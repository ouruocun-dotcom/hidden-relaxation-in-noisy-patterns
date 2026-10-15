"""Regenerate field analyses and junction calculations."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
FIELD=['survival_models','survival_uncertainty','preparation_risks','neutral_modes','field_coordinates','coordinate_comparison','prediction_models','grouped_validation','resolution_analysis','parameter_analysis','field_figures']
JUNCTION=['junction','separatrix_check','noise_on','junction_figure']
def run(name):
    print('RUN '+name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'code'/(name+'.py'))],cwd=ROOT,check=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('part',nargs='?',choices=['all','field','junction'],default='all');args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results'/'matplotlib_cache'))
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    (ROOT/'results'/'figures').mkdir(parents=True,exist_ok=True)
    if args.part in ('all','field'):
        target=ROOT/'results'/'field_data';target.mkdir(parents=True,exist_ok=True)
        for f in (ROOT/'data').iterdir():shutil.copy2(f,target/f.name)
        for name in FIELD:run(name)
    if args.part in ('all','junction'):
        for name in JUNCTION:run(name)
    if (ROOT/'results'/'field_data'/'physical_coordinate_features.json').exists() and (ROOT/'results'/'junction'/'noise_on_results.json').exists():run('diagnostic_figures')
    print('Finished. Outputs: '+str(ROOT/'results'),flush=True)
if __name__=='__main__':main()
