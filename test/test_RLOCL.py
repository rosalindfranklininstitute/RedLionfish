import RedLionfishDeconv.RLDeconv3DReiknaOCL as rlocl

def test_hello_world():
    print ("Hello World!")

def test_get_best_blockshape_and_blockstep():
    #Test a few combinations of data and psf shapes and check result

    blocksizes = [
        256,
        256,
        512,
        256
        ]

    datashapes = [ 
        (512,512,512),
        (256,256,256),
        (834,300,2048),
        (834,300,2048),
        ]

    psfshapes = [
        (32,32,32),
        (32,32,32),
        (93,205,205),
        (93,205,205)
    ]

    expected_results = [
        ( [256, 256, 256],[217, 217, 217] ),
        ( [256, 256, 256],[256, 256, 256] ),
        ( [512, 300, 512],[400, 300, 266] ),
        ( None,None )
    ]
    psfpaddingfract = 1.2

    for ds,ps,bs, exp_res0 in zip(datashapes,psfshapes,blocksizes, expected_results):
        print(f"datashape:{ds}, psfshape:{ps}, blocksize:{bs}")

        bl_sh, bl_st = rlocl.get_best_blockshape_and_blockstep(ds,ps,bs,psfpaddingfract=psfpaddingfract)

        print(f"block_shape:{bl_sh}, block_step:{bl_st}")

        assert bl_sh==exp_res0[0]
        assert bl_st == exp_res0[1]

