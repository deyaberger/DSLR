from classes.description import Describe
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='.csv file containing the data to describe')
    parser.add_argument('-f', '--full_display', help='display all rows of the describe dataframe', action='store_true')
    parser.add_argument('-s', '--save', help='save the info of describe in a csv file', action='store_true')
    parser.add_argument('-q', '--quartile', help='Calculate additional quartiles', action = 'append', type = int, choices = list(range(0, 101)))
    args = parser.parse_args()
    args.list_params = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    if args.quartile:
        for q in args.quartile:
            name = str(q) + "%"
            args.list_params.append(name)
    return (args)

if __name__ == "__main__":
    args = parse_arguments()
    print(args.quartile)
    describe = Describe(args)
    if args.full_display == True:
        print(describe.output_df.to_string())
    else:
        print(describe.output_df)
    if args.save == True:
        describe.output_df.to_csv("describe.csv", index=False)
    