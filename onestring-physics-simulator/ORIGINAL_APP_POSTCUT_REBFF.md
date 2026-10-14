# 元のOneStringアプリで再BFF → T2D Dual Hingeまで実行する

この機能は既存のOptCutsやOneStringシミュレーションを削除・上書きしません。実験ブランチの **app.py** にオプションを追加したもので、従来の `View stage` からM2D / M3D / K3D / T3D / K2D / T2D Top Hinge / T2D Dual Hingeを表示できます。

## Macで実行

**研究用の元フォルダには未コミット作業があるため、実験用worktreeだけを更新**します。

```bash
cd ~/Documents/onestring-postcut-rebff/onestring-physics-simulator
git status --short
git fetch origin
git switch --detach origin/agent/postcut-rebff-mac-20261014
source .venv-postcut-rebff/bin/activate
python3 -m pip install -e .
python3 -m streamlit run app.py
```

変更ファイルがある場合は `git switch` 前に確認してください。既存のBFF実行ファイルは引き続き利用します。矩形CLIが古い場合は `bash scripts/repair_postcut_bff_mac.sh` で再ビルドしてください。

サイドバーの「Paper-oriented computation mode」をONにし、「Split後の矩形BFF再計算 → T2D Dual Hinge（実験）」をONにします。最大Split数を設定し、Sとして開いた三角形メッシュ（例：LiliumのOBJ）を選び、「Run OneString pipeline」を押します。完了したら「View stage」から「T2D Dual Hinge」を選びます。

## CSF未達の警告について

`all_candidate_cuts_failed` は、残っているCSF>2の領域を切る候補を試したが、どの候補も採用条件を満たさなかった、という意味です。例：`Split=1, 領域=2, 最大CSF=7109.1819, stop=all_candidate_cuts_failed` は **CSF条件未達** であり論文再現の成功ではありません。追加切断候補が不適格だった正確な理由は、表示される `rejected_candidates` の各 `reason` を確認してください。典型的には非ディスク領域、退化三角形、矩形BFFの失敗などがありますが、スクリーンショットだけから特定はできません。

CSFが2以下ならそのまま既存のソルバーで最終T2Dへ進みます。CSFが2を超えた場合は、デフォルトでは計算を拒否し棄却履歴を表示します。科学的な比較のため、**「CSF>2で停止してもT2D Dual Hingeの試験計算を許可」** をONにすれば、CSF未達である旨の表示を維持しながら既存ソルバーへ渡します。最終T2Dの出力が得られたとしても製造可能性・元論文との一致・接触衝突のないことを意味しません。下流ソルバーが失敗した場合は例外を表示し、偽のT2Dでは代替しません。

## 内部の流れ

1. Sから初回Ωの矩形BFFを計算。既定ではS境界に合致する四隅候補を自動選択
2. 初回の線形CSFを測定し、≤2であればSplit=0
3. >2の場合は最高曲率点に近いGrid線でΩを切断。元Sも同じ線で切断
4. 2つのS片をそれぞれ矩形境界BFFで展開し直す
5. >2の領域にだけ同じ処理を繰り返す。最大本数・候補なし・全候補失敗で未達停止
6. 複数のUV領域を統合したM2Dを作り、**既存のOneStringのK3D/T3D/K2D/T2D Dual Hinge最適化器**へ渡す

この方式は再BFF後にGridを再生成する研究用処理であり、元のM2Dタイル数を厳密には保存しません。タイル枚数・CSF・T2D評価値は画面で確認してください。既存のOptCuts分岐の挙動は今回の変更対象外です。

## Split境界のT2Dパネル位置合わせ

再BFFした各領域は独立のUV島です。元の実装では島のUVを横方向へ機械的にずらし、さらにT2Dで別々のヒンジ成分を自動的に引き離していたため、物理的に元Sでつながっていた切断境界でも2D表示に大きな隙間ができていました。

現行の実験モードは、元Sの同じ切断境界に由来する3D位置を抽出し、M3Dの対応パネル角を特定します。T2D Dual HingeとTop Hingeの該当領域を**剛体SE(2)変換（回転・平行移動のみ）**して位置合わせします。領域内パネルの厚み・辺長・ヒンジを変形させず、ヒンジ座標を更新して、必要なGap graph・Lift points・String pathも再計算します。

結果の `split_seam_fit_rms_max` と `split_seam_fit_rms_relative_to_tile` は、対応境界点の合わせ込み残差です。元S上に同じ切断点がなければ位置合わせは失敗として停止します。対応関係が乏しい場合も成功の偽装をしません。

**これは縫い目のパネル完全一致を証明するものではありません。** 両側の矩形BFFから別々に再格子化した四角形は、境界上のパネル数・角点位置が一致しない可能性があります。完全一致には、元Ωの共通切断格子を保持するか、両側のBFF後に共通境界頂点・接続対応を制約して再メッシュ化する必要があります。今回の修正は「物理的に対応する境界を使うrigidな2D位置合わせ」であり、形の異なるパネルを引き延ばして一致したふりをするものではありません。

### 更新して表示

```bash
cd ~/Documents/onestring-postcut-rebff/onestring-physics-simulator
git status --short
git fetch origin
git switch --detach origin/agent/postcut-rebff-mac-20261014
source .venv-postcut-rebff/bin/activate
python3 -m streamlit run app.py
```

再BFFを有効にしたまま「Run OneString pipeline」を再実行して、View stage の **T2D Dual Hinge** を確認します。以前の結果がStreamlitセッションに残っていても再計算しない限り新しい配置にはなりません。

## 共通境界を保った再BFF矩形と共有グリッド（追加修正）

- 初回のS→ΩとGrid Splitの後、**実際のSの同一切断線の両端を、切られた2領域のBFF矩形の角として固定**します。もはや各領域の境界周長を独立に4等分して縫い目の位置をずらしません。
- 各BFF出力は一様倍率・回転・平行移動（相似変換）のみで物理的なSplit境界の両端を一致させます。異方性のある伸縮や個々のUV三角形の変形はしないため、計測した線形CSF比は変わりません。
- それらの矩形UVに**共通原点・共通間隔の格子**を敷き、Split境界の両側に同じ座標のM2Dグリッド頂点が2つ以上存在するかを検査します。対応がなければT2Dへの成功として扱いません。
- T2D Dual Hingeの最終レイアウトでは、近傍3D頂点の曖昧な推定に代わり、**共有M2D格子で一致していた同じ頂点**を優先して剛体位置合わせします。その後のGap graph、Lift points、String pathも再生成します。
- `postcut_seam_shared_grid_vertex_count` が正の値、`split_seam_alignment: shared_source_seam_grid_rigid_SE2` であればこの経路が使用されています。`split_seam_fit_rms_relative_to_tile` は最終剛体位置合わせの残差です。
- 注意：共通UV格子の座標が同じでも、独立BFFでのSへの逆写像が縫い目上で完全に同一の3D点を表すとは限りません。**最終T2D全角点の一致と製造可能性を保証するのは、追加の3D縫い目対応・ヒンジ最適化の検証が必要です。**
