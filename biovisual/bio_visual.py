import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

class BioVisual():

    def visual_vec(self,dis_disprot , disprot ,dis_fg_nups , fg_nups , pdb1 , pdb2):
        
        # load 2D vectors
        with open(dis_disprot, "r") as f:
            dis_disprot_vec = pickle.load(f)
        f.close()
        with open(disprot, "r") as f:
            disprot_vec = pickle.load(f)
        f.close()
        with open(dis_fg_nups, "r") as f:
            dis_fg_nups_vec = pickle.load(f)
        f.close()
        with open(fg_nups, "r") as f:
            fg_nups_vec = pickle.load(f)
        f.close()
        with open(pdb1, "r") as f:
            pdb1_vec = pickle.load(f)
        f.close()
        with open(pdb2, "r") as f:
            pdb2_vec = pickle.load(f)
        f.close()
        
        # visualization
        fig , axarr = plt.subplots(2,3, figsize=(15,10))
        
        x = dis_disprot_vec[:,0]
        y = dis_disprot_vec[:,1]
        g1 = axarr[0,0].hist2d(x, y,bins=40 , norm=LogNorm())
        axarr[0,0].set_title("Dis-Disprot")
        # fig.colorbar(g1, ax=axarr[0,0])
        
        x = disprot_vec[:,0]
        y = disprot_vec[:,1]
        g2 = axarr[1,0].hist2d(x, y,bins=40 , norm=LogNorm())
        axarr[1,0].set_title("Disprot")
        # fig.colorbar(g2, ax=axarr[0,1])

        x = pdb1_vec[:,0]
        y = pdb1_vec[:,1]
        g3 = axarr[0,1].hist2d(x, y,bins=40 , norm=LogNorm())
        axarr[0,1].set_title("PDB(random set 1)")
        # fig.colorbar(g3, ax=axarr[0,2])
        
        x = pdb2_vec[:,0]
        y = pdb2_vec[:,1]
        g4 = axarr[1,1].hist2d(x, y,bins=40 , norm=LogNorm())
        axarr[1,1].set_title("PDB(random set 2)")
        # fig.colorbar(g4, ax=axarr[1,0])
    
        x = dis_fg_nups_vec[:,0]
        y = dis_fg_nups_vec[:,1]
        g5 = axarr[0,2].hist2d(x, y,bins=40 , norm=LogNorm())
        axarr[0,2].set_title("Dis-FGNUPS")
        # fig.colorbar(g5, ax=axarr[1,1])
    
        x = fg_nups_vec[:,0]
        y = fg_nups_vec[:,1]
        g6 = axarr[1,2].hist2d(x, y,bins=40 , norm=LogNorm())
        axarr[1,2].set_title("FGNUPS")
        # fig.colorbar(g6, ax=axarr[1,2])

        plt.show()
