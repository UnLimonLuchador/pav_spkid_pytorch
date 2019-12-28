import re
import argparse

def main(opts):
    with open(opts.r) as results:
        with open(opts.r) as results2:
        
            false_negative = 0
            false_positive = 0
            true = 0
            false = 0
            for line in results:        
                aux = re.split("\t",line.rstrip())        
                gt = re.split("/",aux[0])[1]
                clas = aux[1]
                cand = aux[2]
                if cand == gt:
                    true += 1
                    if clas != cand:
                        false_negative += 1
                    
                if cand != gt:
                    false += 1
                    if clas == cand:
                        false_positive += 1
                    

            
            pm=false_negative/true
            pfa=false_positive/false
            rho=0.01
            C=(1/min(rho,1-rho))*(rho*pm+(1-rho)*pfa)*100

            print("False Negative Ratio: ",pm)
            print("False Positive Ratio: ",pfa)
            print("Cost (rho = 0.01): ",C)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NN in verification')    
    parser.add_argument('--r', type=str,
                        help='verification result file') 

    opts = parser.parse_args()
    if opts.r is None:
        raise ValueError('Result file not specified!')
    main(opts)