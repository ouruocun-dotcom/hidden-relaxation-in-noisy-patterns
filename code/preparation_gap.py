"""Model-independent paired risk diagnostic. Input CSV: outcome_a,outcome_b.
Default two-sided bound permits either direction; --directed fixes a-minus-b.
Rows must be independent paired units; within-pair dependence is allowed.
Confidence covers sampling for the supplied fixed endpoint, not numerical error.
"""
import argparse
import csv
import json
import math

def certificate(a,b,alpha=.05,family_size=1,directed=False):
 if len(a)!=len(b) or not a:raise ValueError('nonempty paired outcomes required')
 if any(x not in (0,1) for x in list(a)+list(b)):raise ValueError('binary outcomes required')
 if not 0<alpha<1 or family_size<1:raise ValueError('invalid confidence settings')
 n=len(a);pa=sum(a)/n;pb=sum(b)/n;gap=pa-pb
 radius=math.sqrt(2*math.log((1 if directed else 2)*family_size/alpha)/n)
 lower=max(0,(gap if directed else abs(gap))-radius)
 return {'n':n,'risk_a':pa,'risk_b':pb,'gap':gap,'radius':radius,'gap_magnitude_lower':lower,'common_prediction_error_lower':lower/2,'alpha':alpha,'family_size':family_size,'directed':directed,'scope':'fixed endpoint, independent paired rows; no continuous-time or model error coverage'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv');p.add_argument('--alpha',type=float,default=.05);p.add_argument('--family-size',type=int,default=1);p.add_argument('--directed',action='store_true');args=p.parse_args()
 with open(args.csv,newline='') as f:rows=list(csv.DictReader(f))
 print(json.dumps(certificate([int(r['outcome_a']) for r in rows],[int(r['outcome_b']) for r in rows],args.alpha,args.family_size,args.directed),indent=2))
if __name__=='__main__':main()
