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
