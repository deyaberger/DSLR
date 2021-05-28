from classes import Describe
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='.csv file containing the data to describe')
    parser.add_argument('-f', '--full_display', help='display all rows of the describe dataframe', action='store_true')
    parser.add_argument('-s', '--save', help='save the info of describe in a csv file', action='store_true')
    parser.add_argument('-q', '--quartile', help='Calculate additional quartiles', nargs='+', type = int)
    parser.add_argument('-skw', '--skewness', help='Calculate skewness', action = 'store_true')
    parser.add_argument('-c', '--compare', help="Compare with pandas' describe output", action = 'store_true')
    args = parser.parse_args()
    args.list_params = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    if args.skewness == True:
        args.list_params.append("skw")
    if args.quartile != None:
        for q in args.quartile:
            if q < 0 or q > 100:
                print(f"WARNING: not taking in account invalid quartile {q}")
                continue
            name = str(q) + "%"
            args.list_params.append(name)
    return (args)

if __name__ == "__main__":
    args = parse_arguments()
    describe = Describe(args)
    if args.full_display == True:
        print(describe.output_df.to_string())
    else:
        print(describe.output_df)
    if args.save == True:
        describe.output_df.to_csv("describe.csv", index=False)
    if args.compare == True:
        import pandas as pd
        df = pd.read_csv(args.datafile)
        print("\n\n** Here is the decribe output from the real pandas function to compare: ** \n")
        print(df.describe())
    