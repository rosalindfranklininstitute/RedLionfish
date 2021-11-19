#To be run from the command line
#It needs to be a seperate file otherwise the imports will fail

__version__ = "0.3"

# Ability to run from command line providing tiff (or other) filenames
def main():
    import argparse
    import os
    import sys

    #Argument to showInfo overrides the required arguments
    #The way to achieve this is to create an Action.
    #https://docs.python.org/dev/library/argparse.html#action
    class showInfoAction(argparse.Action):
        # def __init__(self, option_strings, dest, nargs=None, **kwargs):
        #     if nargs is not None:
        #         raise ValueError("nargs not allowed")
        #     super().__init__(option_strings, dest, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlfocl
            if rlfocl.isReiknaAvailable:
                print ("Reikna (OpenCL) is available.")
                #Get GPU info
                rlfocl.printGPUInfo()
            else:
                print("Reikna (OpenCL) is NOT available.")
            parser.exit()

    parser = argparse.ArgumentParser(description="Richardson-Lucy deconvolution of 3D data.")
    argp_group1 = parser.add_argument_group('files', 'RL from files')
    argp_group1.add_argument("data3Dpath", help="Input 3d data file, tiff format")
    argp_group1.add_argument("psf3Dpath" , help="Input 3d psf/otf file, tiff format")
    argp_group1.add_argument("iterations", type=int, default=10, help="Number of iterations")
    argp_group1.add_argument("--outpath" , "-o", help="output filanme")
    argp_group1.add_argument("--method" , help = "force gpu or cpu method. (not implemented yet, automatic, trying gpu first).")

    argp_group2 = parser.add_argument_group('info', 'Various information')
    argp_group2.add_argument("--version", "-v", action="version", version=__version__ , help = "Show version information")
    argp_group2.add_argument("--showinfo", action=showInfoAction , nargs=0, help = "Show GPU information. Useful before high-processing power testing.")

    args= parser.parse_args()

    data3Dpath = args.data3Dpath
    #check files data3Dpath and psf3Dpath exist
    if not os.path.exists(data3Dpath) :
        print(f"File {data3Dpath} could not be found. Exiting.")
        sys.exit()

    psf3Dpath = args.psf3Dpath
    if not os.path.exists(psf3Dpath) :
        print(f"File {psf3Dpath} could not be found. Exiting.")
        sys.exit()

    iterations = args.iterations
    if iterations <=0:
        print(f"Invalid number of iterations {iterations}")
        sys.exit
    
    #setup the filename.
    # if not provided use the data filname with added _it<iterations>.tiff
    outpath = args.outpath
    if outpath is None:
        pathhead, pathtail = os.path.split(args.data3Dpath)
        pathname , ext = os.path.splitext(pathtail)
        outpath = pathname + "_it" + str(iterations) + ".tiff"

    import RedLionfishDeconv as rl
    rl.doRLDeconvolutionFromFiles(data3Dpath, psf3Dpath, iterations, savepath=outpath)

if __name__ == "__main__":
    # Run if called from the command line
    main()


