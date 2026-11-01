import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PhotoCraftBridge:
    """photocraft-cli をPythonから安全に呼び出すラッパー"""

    def __init__(self, cli_path: Optional[str] = None) -> None:
        self.cli_bin = self._resolve_binary(cli_path)

    @staticmethod
    def _resolve_binary(cli_path: Optional[str]) -> Optional[str]:
        if cli_path and Path(cli_path).is_file():
            return str(Path(cli_path).resolve())

        env_bin = os.environ.get("PHOTOCRAFT_CLI")
        if env_bin and Path(env_bin).is_file():
            return str(Path(env_bin).resolve())

        found = shutil.which("photocraft-cli")
        if found:
            return found

        # cargo install 時の標準配置先を確認
        cargo_bin = Path.home() / ".cargo" / "bin"
        for name in ("photocraft-cli", "photocraft-cli.exe"):
            candidate = cargo_bin / name
            if candidate.is_file():
                return str(candidate)

        return None

    @property
    def is_available(self) -> bool:
        return self.cli_bin is not None

    def inspect_document(self, psd_path: Path) -> Optional[Dict[str, Any]]:
        """photocraft-cli info でPSD構造を検証する"""
        if not self.cli_bin:
            return None
        res = subprocess.run(
            [self.cli_bin, "info", str(psd_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"[WARN] photocraft-cli info 失敗: {res.stderr.strip()}")
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return {"raw_output": res.stdout.strip()}

    def render_or_run(
        self,
        psd_path: Path,
        out_png_path: Path,
        commands: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    ) -> bool:
        """
        PSDをPhotoCraftのコンポジタでPNG書き出しする。
        commands 指定時は `photocraft-cli run`、未指定時は `photocraft-cli convert` を実行。
        """
        if not self.cli_bin:
            return False

        if commands:
            args = [self.cli_bin, "run", str(psd_path)]
            for cmd_id, params in commands:
                args.extend(["--cmd", cmd_id, "--params", json.dumps(params)])
            args.extend(["--out", str(out_png_path)])
        else:
            args = [self.cli_bin, "convert", str(psd_path), str(out_png_path)]

        res = subprocess.run(args, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[WARN] photocraft-cli 書き出し失敗: {res.stderr.strip()}")
            return False
        return True

    def search_registry(self, keyword: Optional[str] = None) -> str:
        """実行PC上のPhotoCraftが持つ正確なコマンドID一覧を出力する"""
        if not self.cli_bin:
            return "photocraft-cli が見つかりません。PATHまたは PHOTOCRAFT_CLI 環境変数を確認してください。"
        args = [self.cli_bin, "commands", "--json"]
        if keyword:
            args.extend(["--filter", keyword])
        res = subprocess.run(args, capture_output=True, text=True, check=False)
        return res.stdout if res.returncode == 0 else res.stderr
