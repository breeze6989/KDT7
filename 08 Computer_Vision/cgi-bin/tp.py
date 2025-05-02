import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

##########################
# HOG / HSV 파라미터
##########################
hog_winSize     = (64,64)
hog_blockSize   = (16,16)
hog_blockStride = (8,8)
hog_cellSize    = (8,8)
hog_nbins       = 9

hog_descriptor = cv2.HOGDescriptor(
    _winSize=hog_winSize,
    _blockSize=hog_blockSize,
    _blockStride=hog_blockStride,
    _cellSize=hog_cellSize,
    _nbins=hog_nbins
)

def extract_feature_hsv_hog(img_bgr, h_bins=8, s_bins=8, v_bins=8):
    # HSV hist
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv],[0,1,2],None,[h_bins,s_bins,v_bins],[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)  # shape=(512,)

    # HOG
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hog_vec = hog_descriptor.compute(img_gray)
    hog_vec = hog_vec.flatten().astype(np.float32)

    feat = np.concatenate([hist, hog_vec], axis=0)  # 대략 512 + 1764 = 2276
    return feat

def augment_image(img_bgr):
    """
    간단 증강:
     - 원본
     - 좌우 flip
     - ±15도 회전
    """
    results = []
    results.append(img_bgr)
    # 좌우 flip
    flip_lr = cv2.flip(img_bgr, 1)
    results.append(flip_lr)

    h, w = img_bgr.shape[:2]
    center = (w//2, h//2)

    # +15
    mat_p15 = cv2.getRotationMatrix2D(center, 15, 1.0)
    rot_p15 = cv2.warpAffine(img_bgr, mat_p15, (w,h))
    results.append(rot_p15)

    # -15
    mat_m15 = cv2.getRotationMatrix2D(center, -15, 1.0)
    rot_m15 = cv2.warpAffine(img_bgr, mat_m15, (w,h))
    results.append(rot_m15)

    return results


def train_all_models_from_images_incremental(
    train_csv_path,
    test_csv_path,
    do_augment=True,
    n_splits=5,
    max_iter_logistic=2000,
    pca_dim=256,
    chunk_size=300   # *** 변경: 첫 배치 >=256 필요
):
   
    
    df_train = pd.read_csv(train_csv_path)
    df_test  = pd.read_csv(test_csv_path)

    for c in ['img_path','과']:
        if c not in df_train.columns:
            raise KeyError(f"[오류] 학습CSV에 {c} 없음.")
        if c not in df_test.columns:
            raise KeyError(f"[오류] 테스트CSV에 {c} 없음.")

    print(f"[train_df] {len(df_train)} rows, [test_df] {len(df_test)} rows")

    # Incremental PCA
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=chunk_size)

    # 1) partial_fit
    train_idxs = df_train.index.tolist()
    for start_i in range(0, len(train_idxs), chunk_size):
        end_i = start_i+chunk_size
        sub_idxs = train_idxs[start_i:end_i]

        feats_list = []
        for idx in sub_idxs:
            row = df_train.loc[idx]
            fn = row['img_path']
            img_bgr = cv2.imread(fn)
            if img_bgr is None: 
                continue

            if do_augment:
                imgs = augment_image(img_bgr)
            else:
                imgs = [img_bgr]

            for one_img in imgs:
                feat = extract_feature_hsv_hog(one_img)
                feats_list.append(feat)

        if len(feats_list)==0:
            continue

        feats_chunk = np.array(feats_list, dtype=np.float32)
        # partial_fit
        ipca.partial_fit(feats_chunk)
        print(f"partial_fit batch=({start_i}~{end_i}), shape={feats_chunk.shape}")

    print(f"[IncrementalPCA partial_fit done], n_components={pca_dim}")

    # 2) transform => train set
    X_train_list, y_train_list = [], []
    for start_i in range(0, len(train_idxs), chunk_size):
        end_i = start_i+chunk_size
        sub_idxs = train_idxs[start_i:end_i]

        feats_list2 = []
        label_list2 = []
        for idx in sub_idxs:
            row = df_train.loc[idx]
            fn = row['img_path']
            label = row['과']

            img_bgr = cv2.imread(fn)
            if img_bgr is None:
                continue

            if do_augment:
                imgs = augment_image(img_bgr)
            else:
                imgs = [img_bgr]

            for one_img in imgs:
                feat = extract_feature_hsv_hog(one_img)
                feats_list2.append(feat)
                label_list2.append(label)

        if len(feats_list2)==0:
            continue
        chunk_arr = np.array(feats_list2, dtype=np.float32)
        chunk_pca = ipca.transform(chunk_arr)
        X_train_list.append(chunk_pca)
        y_train_list.extend(label_list2)

    X_train_final = np.concatenate(X_train_list, axis=0) if X_train_list else None
    y_train_final = np.array(y_train_list)

    print(f"[train_df final] => X.shape={X_train_final.shape}, y.shape={y_train_final.shape}")

    # 3) transform => test set
    test_idxs = df_test.index.tolist()
    X_test_list, y_test_list = [], []
    for start_i in range(0, len(test_idxs), chunk_size):
        end_i = start_i+chunk_size
        sub_idxs = test_idxs[start_i:end_i]

        feats_list3 = []
        label_list3 = []
        for idx in sub_idxs:
            row = df_test.loc[idx]
            fn = row['img_path']
            label = row['과']

            img_bgr = cv2.imread(fn)
            if img_bgr is None:
                continue

            # 테스트셋은 증강X
            feats = extract_feature_hsv_hog(img_bgr)
            feats_list3.append(feats)
            label_list3.append(label)

        if len(feats_list3)==0:
            continue
        arr_test = np.array(feats_list3, dtype=np.float32)
        arr_test_pca = ipca.transform(arr_test)
        X_test_list.append(arr_test_pca)
        y_test_list.extend(label_list3)

    X_test_final = np.concatenate(X_test_list, axis=0) if X_test_list else None
    y_test_final = np.array(y_test_list)

    print(f"[test_df final] => X.shape={X_test_final.shape}, y.shape={y_test_final.shape}")

    # 4) Scaling
    scaler = StandardScaler()
    scaler.fit(X_train_final)
    X_train_sc = scaler.transform(X_train_final)
    X_test_sc  = scaler.transform(X_test_final)

    # 5) KFold (학습데이터)
    print("\n=== KFold 교차검증 ===")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_map = {
        'Logistic': LogisticRegression(max_iter=max_iter_logistic),
        'SVC':      SVC(probability=True, kernel='rbf', gamma='scale'),
        'KNN':      KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoost': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for mname, mobj in model_map.items():
        acc_list, f1_list = [], []
        for tr_idx, val_idx in kf.split(X_train_sc):
            X_tr, X_val = X_train_sc[tr_idx], X_train_sc[val_idx]
            y_tr, y_val = y_train_final[tr_idx], y_train_final[val_idx]
            mobj.fit(X_tr, y_tr)
            pv = mobj.predict(X_val)
            a_ = accuracy_score(y_val, pv)
            f_ = f1_score(y_val, pv, average='macro', zero_division=0)
            acc_list.append(a_)
            f1_list.append(f_)
        print(f"[{mname}] => Acc={np.mean(acc_list):.4f}, F1={np.mean(f1_list):.4f}")

    # 6) hold-out
    print("\n=== hold-out 학습 ===")
    from sklearn.model_selection import train_test_split
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(X_train_sc, y_train_final,
                                                    test_size=0.2, random_state=42)
    final_models = {}
    for mname, mobj in model_map.items():
        mobj.fit(X_tr2, y_tr2)
        pval2 = mobj.predict(X_val2)
        a2 = accuracy_score(y_val2, pval2)
        f2 = f1_score(y_val2, pval2, average='macro', zero_division=0)
        print(f"[{mname}] hold-out => acc={a2:.4f}, f1={f2:.4f}")
        final_models[mname] = mobj

    # 7) 테스트셋 평가
    print("\n=== 테스트셋 평가 ===")
    best_model, best_f1, best_name = None, -1.0, None
    for mname, mobj in final_models.items():
        ptest = mobj.predict(X_test_sc)
        a_t = accuracy_score(y_test_final, ptest)
        f_t = f1_score(y_test_final, ptest, average='macro', zero_division=0)
        print(f"- {mname} => Acc={a_t:.4f}, F1={f_t:.4f}")
        if f_t> best_f1:
            best_f1 = f_t
            best_name = mname
            best_model= mobj

    print(f"\n[테스트셋] 가장 좋은 모델 => {best_name} (F1={best_f1:.4f})")
    print("=== 완료 ===")
    return final_models


train_all_models_from_images_incremental('../team_project/labeled_train.csv','../team_project/test.csv')