import numpy as np
from collections import namedtuple

# --- Définition des classes et couleurs (tiré du notebook) ---

Label = namedtuple( 'Label' , ['name', 'id', 'trainId', 'category', 'catId', 'hasInstances', 'ignoreInEval', 'color'])
labels = [
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ), Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ), Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ), Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ), Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ), Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ), Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ), Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ), Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ), Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ), Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ), Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ), Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ), Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ), Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ), Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ), Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ), Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

id_to_catId = {label.id: (0 if label.ignoreInEval else label.catId) for label in labels}
CATEGORY_INFO = {
    0: {'name': 'void', 'color': (0, 0, 0)}, 1: {'name': 'flat', 'color': (128, 64, 128)},
    2: {'name': 'construction', 'color': (70, 70, 70)}, 3: {'name': 'object', 'color': (220, 220, 0)},
    4: {'name': 'nature', 'color': (107, 142, 35)}, 5: {'name': 'sky', 'color': (70, 130, 180)},
    6: {'name': 'human', 'color': (220, 20, 60)}, 7: {'name': 'vehicle', 'color': (0, 0, 142)},
}
N_CLASSES = len(CATEGORY_INFO)

lookup_table = np.zeros(35, dtype=np.uint8)
for i in range(35): 
    lookup_table[i] = id_to_catId.get(i, 0)

def map_mask_to_8_classes(mask):
    """Convertit un masque Cityscapes brut en 8 catégories."""
    clipped_mask = np.clip(mask, 0, 34)
    return lookup_table[clipped_mask]

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque de classe 2D en une image RGB."""
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, info in CATEGORY_INFO.items():
        rgb_mask[mask == class_id] = info['color']
    return rgb_mask

def calculate_iou(gt_mask, pred_mask):
    """Calcule le score d'Intersection over Union (IoU) moyen."""
    # S'assurer que les masques ont la même forme
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Les masques doivent avoir la même dimension pour calculer l'IoU.")
    
    iou_scores = []
    # On ignore la classe 0 (void) dans le calcul du score moyen
    for cls in range(1, N_CLASSES):
        intersection = np.logical_and(gt_mask == cls, pred_mask == cls).sum()
        union = np.logical_or(gt_mask == cls, pred_mask == cls).sum()
        
        if union == 0:
            # Si la classe n'est ni dans la prédiction ni dans la vérité terrain, l'IoU est parfait (1.0)
            # ou on peut l'ignorer. L'ignorer est plus courant.
            continue
        
        iou = intersection / union
        iou_scores.append(iou)
        
    # Retourner la moyenne des scores IoU des classes présentes
    return np.mean(iou_scores) if iou_scores else 0.0