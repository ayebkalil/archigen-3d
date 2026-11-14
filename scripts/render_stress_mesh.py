"""Render sampled mesh projections for review, without claiming scan-based fidelity.

Use the main environment (matplotlib, trimesh) on one saved run directory.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import trimesh
    scene = trimesh.load(args.directory/'model.glb', force='scene', process=False)
    mesh = scene.to_geometry()
    faces = mesh.faces[::max(1, int(np.ceil(len(mesh.faces)/20000)))]
    triangles = mesh.vertices[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
    shade = .35+.6*np.abs(normals @ np.array([.3,.8,.5]))
    colors = np.clip(np.column_stack((shade*.8,shade*.85,shade)),0,1)
    fig = plt.figure(figsize=(11,4),layout='constrained')
    bounds = mesh.bounds; center=bounds.mean(axis=0); half=max(mesh.extents)/2
    for index, azimuth in enumerate((-60,30,120)):
        ax=fig.add_subplot(1,3,index+1,projection='3d')
        # Plot coordinates x,z,y so the project's Y-up model is upright.
        ax.add_collection3d(Poly3DCollection(triangles[:,:,[0,2,1]],facecolors=colors,linewidths=0))
        ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[2]-half,center[2]+half),
               zlim=(center[1]-half,center[1]+half),xlabel='x (m)',ylabel='z (m)',zlabel='y (m)')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=azimuth)
    fig.suptitle(f'Mesh projections: {len(faces):,}/{len(mesh.faces):,} faces; geometry only, texture not shown')
    fig.savefig(args.directory/'mesh-views.png',dpi=110);plt.close(fig)
    for item in scene.geometry.values():
        texture=getattr(getattr(item.visual,'material',None),'baseColorTexture',None)
        if texture is not None:
            texture.save(args.directory/'atlas.png');break
    print(args.directory/'mesh-views.png')


if __name__=='__main__':main()
