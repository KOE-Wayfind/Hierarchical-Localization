from matplotlib import cm
import random
import numpy as np
import pickle
import pycolmap

from .utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from .utils.io import read_image


# Taken from visualization.py
def extract_loc_from_log(image_dir, query_name, loc, reconstruction=None,
                           db_image_dir=None, top_k_db=2, dpi=75):


    result = []

    q_image = read_image(image_dir / query_name)
    if loc.get('covisibility_clustering', False):
        # select the first, largest cluster if the localization failed
        loc = loc['log_clusters'][loc['best_cluster'] or 0]

    inliers = np.array(loc['PnP_ret']['inliers'])
    mkp_q = loc['keypoints_query']
    n = len(loc['db'])
    if reconstruction is not None:
        # for each pair of query keypoint and its matched 3D point,
        # we need to find its corresponding keypoint in each database image
        # that observes it. We also count the number of inliers in each.
        kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
        counts = np.zeros(n)
        dbs_kp_q_db = [[] for _ in range(n)]
        inliers_dbs = [[] for _ in range(n)]
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                         kp_to_3D_to_db)):
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            for db_idx in db_idxs:
                counts[db_idx] += inl
                kp_db = track[loc['db'][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    else:
        # for inloc the database keypoints are already in the logs
        assert 'keypoints_db' in loc
        assert 'indices_db' in loc
        counts = np.array([
            np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            db = reconstruction.images[loc['db'][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            kp_q = mkp_q[db_kp_q_db[:, 0]]
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            db_name = loc['db'][db_idx]
            kp_q = mkp_q[loc['indices_db'] == db_idx]
            kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
            inliers_db = inliers[loc['indices_db'] == db_idx]

        db_image = read_image((db_image_dir or image_dir) / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f'inliers: {sum(inliers_db)}/{len(inliers_db)}'

        print("db_image", db_image)
        print("db_name", db_name)

        result.add(db_name)


        # plot_images([q_image, db_image], dpi=dpi)
        # plot_matches(kp_q, kp_db, color, a=0.1)
        # add_text(0, text)
        # opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        # add_text(0, query_name, **opts)
        # add_text(1, db_name, **opts)
    return result
