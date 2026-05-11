"""VTplot.py — 振動試験データ汎用プロットツール

DataKind / Frequency / X-AxisUnit / Y-AxisUnit ヘッダーからグラフ種を自動判定し、
PSD / FRF(Real/Imag/Mag/Phase) / Coherence を一つのウィンドウで切り替えてプロット。
共振解析機能付き（scipy 使用）。
"""
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import TextBox, Button

try:
    from scipy.signal import find_peaks as _sp_find_peaks
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ──────────────────────────────────────────────────────────────────────────────
# ファイル解析
# ──────────────────────────────────────────────────────────────────────────────

def parse_vt_file(filepath):
    meta = {}
    freqs, values = [], []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts   = [p.strip() for p in line.split(',')]
        key_raw = parts[0]
        rest    = parts[1:]
        if key_raw.endswith(':') and rest:
            key = key_raw[:-1]
            if key == 'DataKind':
                meta['channel']    = rest[0] if len(rest) > 0 else ''
                meta['graph_type'] = rest[1] if len(rest) > 1 else ''
                meta['sub_type']   = rest[2] if len(rest) > 2 else ''
            elif key == 'Frequency':
                try:
                    meta['freq_min'] = float(rest[0])
                    meta['freq_max'] = float(rest[1])
                except (ValueError, IndexError):
                    pass
            elif key == 'X-AxisUnit':
                meta['x_unit'] = rest[0]
            elif key == 'Y-AxisUnit':
                meta['y_unit'] = rest[0]
            elif key == 'X-AxisScale':
                meta['x_scale_hint'] = rest[0]
            elif key == 'Y-AxisScale':
                meta['y_scale_hint'] = rest[0]
            continue
        if len(parts) >= 2:
            try:
                freqs.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                pass
    return meta, np.array(freqs), np.array(values)


def classify(meta):
    g = meta.get('graph_type', '').lower()
    s = meta.get('sub_type',   '').lower()
    if 'powerspec' in g:
        return 'PSD'
    if 'frf' in g:
        for sub, label in [('real',  'FRF_Real'), ('imag',  'FRF_Imag'),
                            ('phase', 'FRF_Phase'), ('mag',  'FRF_Mag')]:
            if sub in s:
                return label
        return 'FRF_Mag'
    if 'coherence' in g or g == 'coh':
        return 'Coherence'
    return 'Unknown'


KIND_LABEL = {
    'PSD':       'PSD',
    'FRF_Real':  'FRF Real',
    'FRF_Imag':  'FRF Imag',
    'FRF_Mag':   'FRF Mag',
    'FRF_Phase': 'FRF Phase',
    'Coherence': 'Coherence',
}
KIND_ORDER = ['PSD', 'FRF_Real', 'FRF_Imag', 'FRF_Mag', 'FRF_Phase', 'Coherence']

DEFAULT_YSCALE = {
    'PSD':       'log',
    'FRF_Real':  'linear',
    'FRF_Imag':  'linear',
    'FRF_Mag':   'log',
    'FRF_Phase': 'linear',
    'Coherence': 'linear',
}

_COL_FREQ_IDLE   = '#bbbbbb'
_COL_FREQ_ACTIVE = '#1155cc'
_COL_FREQ_LOCKED = '#cc4400'
_STARS = ['', '★', '★★', '★★★', '★★★★', '★★★★★']


# ──────────────────────────────────────────────────────────────────────────────
# 共振解析
# ──────────────────────────────────────────────────────────────────────────────

def find_resonances(all_data, freq_range=None):
    """全チャンネルの共振候補を検出し、スコア降順のリストを返す。
    freq_range=(fmin, fmax) を渡すと表示範囲内のみ解析する。
    scipy がない場合は None を返す。
    """
    if not _SCIPY:
        return None

    # indexed[ch][kind] = (freqs, vals, color)
    indexed = {}
    for kind, entries in all_data.items():
        for _, freqs, vals, ch, color in entries:
            indexed.setdefault(ch, {})[kind] = (freqs, vals, color)

    results = []

    for ch, kinds in indexed.items():
        # 主指標の選択
        if 'FRF_Mag' in kinds:
            primary = 'FRF_Mag'
        elif 'PSD' in kinds:
            primary = 'PSD'
        else:
            continue

        freqs_all, vals_all, color = kinds[primary]

        # 表示周波数範囲でフィルタ
        if freq_range is not None:
            fmin_r, fmax_r = freq_range
            mask = (freqs_all >= fmin_r) & (freqs_all <= fmax_r)
            freqs, vals = freqs_all[mask], vals_all[mask]
        else:
            freqs, vals = freqs_all, vals_all

        if len(freqs) < 10:
            continue

        # DC 除外・正値のみ
        valid = (vals > 0) & (freqs > 0)
        if np.sum(valid) < 5:
            continue

        log_v      = np.where(valid, np.log10(vals), np.nan)
        log_min    = float(np.nanmin(log_v)) - 1.0
        log_filled = np.where(np.isnan(log_v), log_min, log_v)

        n        = len(freqs)
        min_dist = max(3, n // 200)

        try:
            peaks, props = _sp_find_peaks(log_filled, prominence=0.3, distance=min_dist)
        except Exception:
            continue

        if len(peaks) == 0:
            continue

        # prominence 降順で上位 15 件
        order = np.argsort(props['prominences'])[::-1][:15]

        for oi in order:
            pi   = int(peaks[oi])
            prom = float(props['prominences'][oi])
            freq = float(freqs[pi])
            mag  = float(vals[pi])

            # 背景レベルとの比率
            win  = max(10, n // 15)
            lo   = max(0, pi - win)
            hi   = min(n, pi + win + 1)
            nbrs = np.concatenate([vals[lo:pi], vals[pi+1:hi]])
            nbrs = nbrs[nbrs > 0]
            bg   = float(np.median(nbrs)) if len(nbrs) else mag
            ratio = mag / bg if bg > 0 else 1.0
            if ratio < 1.5:
                continue

            ev    = []   # evidence lines
            score = 0

            # ── 基本スコア（ピーク比） ──────────────────────────────────
            stars_base = '★★' if ratio >= 10 else '★'
            if ratio >= 10:
                score += 2
            else:
                score += 1
            ev.append(f'{primary} ピーク: {mag:.3e}  (背景の {ratio:.1f} 倍) {stars_base}')

            # ── FRF Phase ──────────────────────────────────────────────
            if 'FRF_Phase' in kinds:
                pf, pv, _ = kinds['FRF_Phase']
                ph       = float(np.interp(freq, pf, pv))
                near90   = abs(abs(ph) - 90) < 45
                srch     = max(3, int(len(pf) * 0.025))
                ci       = int(np.searchsorted(pf, freq))
                i0, i1   = max(0, ci - srch), min(len(pf) - 1, ci + srch)
                ph_chg   = abs(float(np.interp(pf[i1], pf, pv)) -
                               float(np.interp(pf[i0], pf, pv)))
                cross    = ph_chg > 60
                notes    = []
                if near90: notes.append('±90°付近')
                if cross:  notes.append(f'位相変化 {ph_chg:.0f}°')
                note_s   = ' (' + ', '.join(notes) + ' ✓)' if notes else ''
                ev.append(f'FRF Phase  {ph:.1f}° @ {freq:.1f} Hz{note_s}')
                if near90 and cross:
                    score += 2
                elif near90 or cross:
                    score += 1

            # ── FRF Imaginary ──────────────────────────────────────────
            if 'FRF_Imag' in kinds:
                imf, imv, _ = kinds['FRF_Imag']
                std_im = float(np.std(imv))
                if std_im > 0:
                    ip_pos, _ = _sp_find_peaks( imv, prominence=std_im * 0.3)
                    ip_neg, _ = _sp_find_peaks(-imv, prominence=std_im * 0.3)
                    ip_all    = np.concatenate([ip_pos, ip_neg])
                    if len(ip_all):
                        nf  = float(imf[ip_all[np.argmin(np.abs(imf[ip_all] - freq))]])
                        hit = abs(nf - freq) / (freq + 1e-9) < 0.05
                        ev.append(f'FRF Imag   ピーク: {nf:.1f} Hz {"✓" if hit else "(5%超)"}')
                        if hit:
                            score += 1

            # ── FRF Real ───────────────────────────────────────────────
            if 'FRF_Real' in kinds:
                rf, rv, _ = kinds['FRF_Real']
                ci2  = int(np.searchsorted(rf, freq))
                srch2 = max(3, int(len(rf) * 0.025))
                seg  = rv[max(0, ci2 - srch2):min(len(rv), ci2 + srch2 + 1)]
                zc   = bool(np.any(np.diff(np.sign(seg)) != 0))
                ev.append(f'FRF Real   ゼロクロス @ {freq:.1f} Hz {"✓" if zc else "×"}')
                if zc:
                    score += 1

            # ── Coherence ──────────────────────────────────────────────
            if 'Coherence' in kinds:
                cf, cv2, _ = kinds['Coherence']
                coh = float(np.clip(np.interp(freq, cf, cv2), 0, 1))
                q   = '高' if coh >= 0.9 else '中' if coh >= 0.7 else '低'
                ev.append(f'Coherence  {coh:.3f} @ {freq:.1f} Hz (信頼性: {q})'
                           f'{"  ✓" if coh >= 0.8 else ""}')
                if coh >= 0.8:
                    score += 1

            # ── PSD 照合（FRF が主指標の場合） ─────────────────────────
            if primary != 'PSD' and 'PSD' in all_data:
                for _, pf2, pv2, pch, _ in all_data['PSD']:
                    good = pv2 > 0
                    if not np.any(good):
                        continue
                    lv2  = np.where(good, np.log10(pv2),
                                    float(np.nanmin(np.log10(pv2[good]))) - 1)
                    pp, _ = _sp_find_peaks(lv2, prominence=0.2)
                    if not len(pp):
                        continue
                    nf2  = float(pf2[pp[np.argmin(np.abs(pf2[pp] - freq))]])
                    if abs(nf2 - freq) / (freq + 1e-9) < 0.05:
                        ev.append(f'PSD ({pch}) ピーク一致: {nf2:.1f} Hz ✓')
                        score += 1
                        break

            # ── 半値帯域幅（-3 dB） ────────────────────────────────────
            hp = mag / np.sqrt(2)
            f1 = f2 = None
            for i in range(pi - 1, -1, -1):
                if vals[i] <= hp:
                    d  = vals[i + 1] - vals[i]
                    f1 = (freqs[i] + (hp - vals[i]) / d * (freqs[i + 1] - freqs[i])
                          if abs(d) > 1e-30 else freqs[i])
                    break
            for i in range(pi + 1, len(vals)):
                if vals[i] <= hp:
                    d  = vals[i] - vals[i - 1]
                    f2 = (freqs[i - 1] + (hp - vals[i - 1]) / d * (freqs[i] - freqs[i - 1])
                          if abs(d) > 1e-30 else freqs[i])
                    break
            if f1 is not None and f2 is not None and f2 > f1:
                bw   = float(f2 - f1)
                zeta = bw / (2 * freq) * 100
                ev.append(f'半値帯域幅  {bw:.2f} Hz (-3 dB) → 減衰比 ζ = {zeta:.2f} %')

            results.append({
                'freq':       freq,
                'ch':         ch,
                'primary':    primary,
                'score':      score,
                'prominence': prom,
                'ratio':      ratio,
                'color':      color,
                'evidence':   ev,
            })

    results.sort(key=lambda x: (x['score'], x['prominence']), reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── ファイル選択 ───────────────────────────────────────────────────────────
    _sel = tk.Tk()
    _sel.withdraw()
    paths = filedialog.askopenfilenames(
        title='振動データファイルを選択（複数可）',
        filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
    )
    _sel.destroy()
    if not paths:
        print('ファイルが選択されませんでした。')
        return

    # ── データ読み込み・分類 ───────────────────────────────────────────────────
    cmap      = plt.get_cmap('tab10')
    color_idx = {}
    all_data  = {}

    for fp in sorted(paths):
        meta, freqs, values = parse_vt_file(fp)
        if freqs.size == 0:
            continue
        kind  = classify(meta)
        ch    = meta.get('channel', '?')
        if ch not in color_idx:
            color_idx[ch] = len(color_idx)
        color = cmap(color_idx[ch] % 10)
        all_data.setdefault(kind, []).append((meta, freqs, values, ch, color))

    if not all_data:
        print('有効なデータが見つかりませんでした。')
        return

    avail_kinds  = [k for k in KIND_ORDER if k in all_data]
    all_channels = list(color_idx.keys())
    ch_colors    = {ch: cmap(color_idx[ch] % 10) for ch in all_channels}

    # ── 状態変数 ────────────────────────────────────────────────────────────────
    cur_kind         = [avail_kinds[0]]
    xscale           = {k: 'log'                           for k in avail_kinds}
    yscale           = {k: DEFAULT_YSCALE.get(k, 'linear') for k in avail_kinds}
    ch_visible       = {ch: True for ch in all_channels}
    axis_ranges      = {k: {'xmin': '', 'xmax': '', 'ymin': '', 'ymax': ''}
                        for k in avail_kinds}
    plot_lines       = {}
    cursor_vline     = [None]
    cursor_locked    = [False]
    locked_freq      = [None]
    resonance_results = [None]   # 共振解析結果（None = 未実行）

    ip_state = {
        'freq_disp': None,
        'lock_disp': None,
        'cb_patch':  {},
        'ch_name_t': {},
        'ch_val_t':  {},
        'cur_chs':   [],
        'RT': 0.900,
        'RH': 0.1,
    }

    # ── tkinter メインウィンドウ ──────────────────────────────────────────────
    root = tk.Tk()
    root.title('VTplot')

    top_bar = tk.Frame(root, bg='#d0d8e0', pady=5)
    top_bar.pack(side=tk.TOP, fill=tk.X)
    tk.Label(top_bar, text='  Graph Type :', bg='#d0d8e0',
             font=('Arial', 10)).pack(side=tk.LEFT)
    kind_var = tk.StringVar(value=KIND_LABEL[avail_kinds[0]])
    combo = ttk.Combobox(top_bar, textvariable=kind_var,
                         values=[KIND_LABEL[k] for k in avail_kinds],
                         state='readonly', width=16, font=('Arial', 10))
    combo.pack(side=tk.LEFT, padx=6)

    ttk.Separator(top_bar, orient='vertical').pack(side=tk.LEFT, padx=10, fill='y', pady=3)

    btn_res = tk.Button(top_bar, text='共振解析', bg='#d0c4e8',
                        activebackground='#b8a4d4', font=('Arial', 10))
    btn_res.pack(side=tk.LEFT, padx=4)
    btn_ann_clear = tk.Button(top_bar, text='アノテーション削除', bg='#e8dcc4',
                              activebackground='#d4c4a4', font=('Arial', 9))
    btn_ann_clear.pack(side=tk.LEFT, padx=2)

    # Figure
    fig = plt.figure(figsize=(18, 8.8))
    fig.patch.set_facecolor('#e0e0e0')
    canvas = FigureCanvasTkAgg(fig, master=root)

    toolbar_frame = tk.Frame(root, bg='#e8e8e8')
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── メインプロット ────────────────────────────────────────────────────────
    ax = fig.add_axes([0.055, 0.10, 0.595, 0.86])
    ax.set_facecolor('#ffffff')

    PX = 0.663
    PW = 0.325

    IP = fig.add_axes([PX, 0.482, PW, 0.500], frameon=True)
    IP.set_facecolor('#f8f8f8')
    IP.set_xticks([]); IP.set_yticks([])
    IP.set_xlim(0, 1); IP.set_ylim(0, 1)
    for sp in IP.spines.values():
        sp.set_edgecolor('#aaaaaa')

    AR = fig.add_axes([PX, 0.177, PW, 0.292], frameon=True)
    AR.set_facecolor('#f8f8f8')
    AR.set_xticks([]); AR.set_yticks([])
    for sp in AR.spines.values():
        sp.set_edgecolor('#aaaaaa')
    AR.text(0.5, 0.975, 'Axis Range & Scale', transform=AR.transAxes,
            ha='center', va='top', fontsize=9, fontweight='bold', color='#333')

    BH  = 0.034
    BX1 = PX + 0.008
    BW  = (PW - 0.030) / 2
    BX2 = BX1 + BW + 0.014
    DX  = BX1 + BW + 0.007
    TW  = (BW - 0.006) / 2

    fig.text(BX1, 0.430, 'X range (Hz)', fontsize=8.5, va='center', color='#444')
    tb_xmin = TextBox(fig.add_axes([BX1, 0.386, BW, BH]), '', initial='', textalignment='center')
    fig.text(DX,  0.403, '–', fontsize=12, ha='center', va='center', color='#666')
    tb_xmax = TextBox(fig.add_axes([BX2, 0.386, BW, BH]), '', initial='', textalignment='center')

    fig.text(BX1, 0.364, 'Y range  (blank = auto)', fontsize=8.5, va='center', color='#444')
    tb_ymin = TextBox(fig.add_axes([BX1, 0.320, BW, BH]), '', initial='', textalignment='center')
    fig.text(DX,  0.337, '–', fontsize=12, ha='center', va='center', color='#666')
    tb_ymax = TextBox(fig.add_axes([BX2, 0.320, BW, BH]), '', initial='', textalignment='center')

    fig.text(BX1, 0.304, 'X scale', fontsize=8, va='center', color='#444')
    fig.text(BX2, 0.304, 'Y scale', fontsize=8, va='center', color='#444')
    btn_xlog = Button(fig.add_axes([BX1,          0.266, TW, BH]), 'Log', color='#ddd', hovercolor='#bbb')
    btn_xlin = Button(fig.add_axes([BX1+TW+0.006, 0.266, TW, BH]), 'Lin', color='#ddd', hovercolor='#bbb')
    btn_ylog = Button(fig.add_axes([BX2,          0.266, TW, BH]), 'Log', color='#ddd', hovercolor='#bbb')
    btn_ylin = Button(fig.add_axes([BX2+TW+0.006, 0.266, TW, BH]), 'Lin', color='#ddd', hovercolor='#bbb')

    btn_apply = Button(fig.add_axes([BX1, 0.214, BW, BH]), 'Apply',   color='#c4dcf2', hovercolor='#88b8e8')
    btn_save  = Button(fig.add_axes([BX2, 0.214, BW, BH]), 'Save PNG', color='#c4f2c4', hovercolor='#88d888')

    # ── IPパネル再構築 ────────────────────────────────────────────────────────
    def _rebuild_ch_panel():
        IP.cla()
        IP.set_facecolor('#f8f8f8')
        IP.set_xticks([]); IP.set_yticks([])
        IP.set_xlim(0, 1); IP.set_ylim(0, 1)
        for sp in IP.spines.values():
            sp.set_edgecolor('#aaaaaa')
        TA = IP.transAxes

        IP.text(0.04, 0.988, 'Channels & Cursor', ha='left', va='top',
                fontsize=9, fontweight='bold', color='#333', transform=TA)
        ip_state['freq_disp'] = IP.text(
            0.96, 0.988, '—',
            ha='right', va='top', fontsize=10, fontweight='bold',
            color=_COL_FREQ_IDLE, transform=TA)
        ip_state['lock_disp'] = IP.text(
            0.96, 0.963, '', ha='right', va='top',
            fontsize=8, color='#cc4400', fontweight='bold', transform=TA)
        IP.plot([0.01, 0.99], [0.942, 0.942],
                color='#cccccc', lw=0.8, transform=TA, clip_on=False)
        IP.text(0.11, 0.952, 'Ch',       ha='left',  va='center', fontsize=6.5, color='#888', transform=TA)
        IP.text(0.97, 0.952, '@ Cursor', ha='right', va='center', fontsize=6.5, color='#888', transform=TA)

        k = cur_kind[0]
        seen, cur_chs = set(), []
        for _, _, _, ch, _ in all_data.get(k, []):
            if ch not in seen:
                seen.add(ch); cur_chs.append(ch)

        N  = len(cur_chs)
        RT, RB = 0.935, 0.015
        RH = (RT - RB) / max(N, 1)
        ip_state.update({'cur_chs': cur_chs, 'RT': RT, 'RH': RH,
                         'cb_patch': {}, 'ch_name_t': {}, 'ch_val_t': {}})

        for i, ch in enumerate(cur_chs):
            cy    = RT - (i + 0.5) * RH
            color = ch_colors[ch]
            ck_h  = RH * 0.62
            vis   = ch_visible.get(ch, True)
            rect  = mpatches.Rectangle(
                (0.030, cy - ck_h / 2), 0.050, ck_h,
                transform=TA,
                facecolor=color if vis else '#cccccc',
                edgecolor='#333' if vis else '#aaa',
                lw=0.8, clip_on=False)
            IP.add_patch(rect)
            ip_state['cb_patch'][ch]  = rect
            ip_state['ch_name_t'][ch] = IP.text(
                0.105, cy, ch, ha='left', va='center',
                fontsize=7.5, color='#111', transform=TA, alpha=1.0 if vis else 0.35)
            ip_state['ch_val_t'][ch]  = IP.text(
                0.97, cy, '—', ha='right', va='center',
                fontsize=7.5, family='monospace', color=color, transform=TA,
                alpha=1.0 if vis else 0.35)
            if i < N - 1:
                IP.plot([0.02, 0.98], [RT - (i + 1) * RH] * 2,
                        color='#e4e4e4', lw=0.5, transform=TA, clip_on=False)

    # ── 軸範囲 保存・復元 ────────────────────────────────────────────────────
    def _save_ranges():
        k = cur_kind[0]
        axis_ranges[k] = {
            'xmin': tb_xmin.text, 'xmax': tb_xmax.text,
            'ymin': tb_ymin.text, 'ymax': tb_ymax.text,
        }

    def _load_ranges():
        r = axis_ranges[cur_kind[0]]
        tb_xmin.set_val(r['xmin']); tb_xmax.set_val(r['xmax'])
        tb_ymin.set_val(r['ymin']); tb_ymax.set_val(r['ymax'])

    # ── スケールボタン ハイライト ────────────────────────────────────────────
    def _update_scale_buttons():
        k      = cur_kind[0]
        xs, ys = xscale[k], yscale[k]
        for btn, active in [(btn_xlog, xs == 'log'),    (btn_xlin, xs == 'linear'),
                            (btn_ylog, ys == 'log'),    (btn_ylin, ys == 'linear')]:
            c = '#88b8e8' if active else '#ddd'
            btn.color = c
            btn.ax.set_facecolor(c)

    # ── アノテーション描画 ───────────────────────────────────────────────────
    def _draw_annotations():
        res = resonance_results[0]
        if not res:
            return
        xmin, xmax = ax.get_xlim()
        for idx, r in enumerate(res, 1):
            f = r['freq']
            if not (xmin <= f <= xmax):
                continue
            col   = r['color']
            alpha = min(0.18 + r['score'] * 0.07, 0.75)
            ax.axvline(x=f, color=col, alpha=alpha, lw=1.0, ls=':', zorder=3)
            # 周波数ラベル（x=データ座標, y=軸座標）
            ax.text(f, 0.98, f'[{idx}] {f:.0f}Hz',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6.5,
                    color=col, rotation=90, alpha=0.85,
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor='white', alpha=0.55, edgecolor='none'))

    # ── 再描画 ───────────────────────────────────────────────────────────────
    def _redraw():
        ax.cla()
        ax.set_facecolor('#ffffff')
        plot_lines.clear()

        k    = cur_kind[0]
        data = all_data.get(k, [])

        for _, freqs, vals, ch, color in data:
            vis  = ch_visible.get(ch, True)
            ln,  = ax.plot(freqs, vals, color=color, lw=0.9, label=ch, visible=vis)
            plot_lines[ch] = ln

        cursor_vline[0] = ax.axvline(
            x=100, color='#e05000', lw=1.0, ls='--', visible=False, zorder=5)

        xs = xscale[k]
        ys = yscale[k]
        ax.set_xscale(xs)
        if ys == 'log':
            for _, _, vals, _, _ in data:
                if len(vals) and np.any(vals <= 0):
                    ys = 'linear'; yscale[k] = 'linear'; break
        ax.set_yscale(ys)

        meta0  = data[0][0] if data else {}
        x_unit = meta0.get('x_unit', 'Hz')
        y_unit = meta0.get('y_unit', '')
        sub    = meta0.get('sub_type', '')
        y_lbl  = f'{sub} ({y_unit})' if y_unit else sub or 'Value'
        ax.set_xlabel(f'Frequency ({x_unit})', fontsize=11)
        ax.set_ylabel(y_lbl, fontsize=11)
        ax.set_title(KIND_LABEL.get(k, k), fontsize=13, pad=8)
        ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)

        f_min = min((m.get('freq_min', 0)    for m, _, _, _, _ in data), default=0)
        f_max = max((m.get('freq_max', 4000) for m, _, _, _, _ in data), default=4000)
        try:
            xmn = float(tb_xmin.text) if tb_xmin.text.strip() else (f_min if f_min > 0 else 1.0)
            xmx = float(tb_xmax.text) if tb_xmax.text.strip() else f_max
            if xmn > 0 and xmx > xmn:
                ax.set_xlim(xmn, xmx)
        except ValueError:
            pass
        try:
            ymn = float(tb_ymin.text) if tb_ymin.text.strip() else None
            ymx = float(tb_ymax.text) if tb_ymax.text.strip() else None
            if ymn is not None and ymx is not None and ymx > ymn:
                ax.set_ylim(ymn, ymx)
            elif ymn is not None:
                ax.set_ylim(bottom=ymn)
            elif ymx is not None:
                ax.set_ylim(top=ymx)
        except ValueError:
            pass

        _update_scale_buttons()
        _draw_annotations()      # 共振マーカーを重ねる
        canvas.draw_idle()

    # ── カーソル処理 ─────────────────────────────────────────────────────────
    def _set_freq_disp(text, color):
        fd = ip_state['freq_disp']
        if fd:
            fd.set_text(text); fd.set_color(color)

    def _cursor_update(xpos):
        vl = cursor_vline[0]
        if vl is not None:
            vl.set_xdata([xpos, xpos]); vl.set_visible(True)
        col = _COL_FREQ_LOCKED if cursor_locked[0] else _COL_FREQ_ACTIVE
        _set_freq_disp(f'{xpos:.2f} Hz', col)
        k    = cur_kind[0]
        data = all_data.get(k, [])
        for meta, freqs, vals, ch, _ in data:
            cv = ip_state['ch_val_t'].get(ch)
            if cv is None:
                continue
            if ch in plot_lines and plot_lines[ch].get_visible() and len(freqs):
                v = float(np.interp(xpos, freqs, vals))
                u = meta.get('y_unit', '')
                cv.set_text(f'{v:.3e}{(" " + u) if u else ""}')
            else:
                cv.set_text('—')

    def on_motion(event):
        if cursor_locked[0]:
            return
        if event.inaxes is not ax or event.xdata is None or event.xdata <= 0:
            vl = cursor_vline[0]
            if vl is not None:
                vl.set_visible(False)
            _set_freq_disp('move cursor over plot', _COL_FREQ_IDLE)
            canvas.draw_idle()
            return
        _cursor_update(event.xdata)
        canvas.draw_idle()

    def _toggle_ch(ch):
        vis = not ch_visible.get(ch, True)
        ch_visible[ch] = vis
        if ch in plot_lines:
            plot_lines[ch].set_visible(vis)
        color = ch_colors[ch]
        patch = ip_state['cb_patch'].get(ch)
        if patch:
            patch.set_facecolor(color if vis else '#cccccc')
            patch.set_edgecolor('#333' if vis else '#aaa')
        a = 1.0 if vis else 0.35
        nt = ip_state['ch_name_t'].get(ch)
        vt = ip_state['ch_val_t'].get(ch)
        if nt: nt.set_alpha(a)
        if vt: vt.set_alpha(a)
        canvas.draw_idle()

    def on_click(event):
        if event.button != 1:
            return
        if event.inaxes is ax and event.xdata is not None and event.xdata > 0:
            ld = ip_state['lock_disp']
            if cursor_locked[0]:
                cursor_locked[0] = False; locked_freq[0] = None
                if ld: ld.set_text('')
                vl = cursor_vline[0]
                if vl is not None: vl.set_visible(False)
                _set_freq_disp('move cursor over plot', _COL_FREQ_IDLE)
            else:
                cursor_locked[0] = True; locked_freq[0] = event.xdata
                _cursor_update(event.xdata)
                if ld: ld.set_text('LOCKED  click to release')
            canvas.draw_idle()
            return
        if event.inaxes is IP:
            _, ya  = IP.transAxes.inverted().transform((event.x, event.y))
            RT, RH = ip_state['RT'], ip_state['RH']
            for i, ch in enumerate(ip_state['cur_chs']):
                if RT - (i + 1) * RH <= ya <= RT - i * RH:
                    _toggle_ch(ch)
                    return

    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event',  on_click)

    # ── 共振解析 結果ウィンドウ ───────────────────────────────────────────────
    def _show_result_window(res, freq_range=None):
        rng_str = (f'  [{freq_range[0]:.1f} – {freq_range[1]:.1f} Hz]'
                   if freq_range is not None else '')
        win = tk.Toplevel(root)
        win.title(f'共振解析結果  ({len(res)} 件検出){rng_str}')
        win.geometry('620x560')

        # テキストエリア
        txt = tk.Text(win, font=('Consolas', 9), wrap=tk.WORD, padx=10, pady=8,
                      bg='#fafafa', relief=tk.FLAT)
        sbar = tk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sbar.set)
        sbar.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(fill=tk.BOTH, expand=True)

        # タグ定義
        txt.tag_configure('title',   font=('Consolas', 11, 'bold'), foreground='#1144aa')
        txt.tag_configure('sep',     foreground='#aaaaaa')
        txt.tag_configure('head',    font=('Consolas', 10, 'bold'), foreground='#222')
        txt.tag_configure('ev',      font=('Consolas', 9),  foreground='#444')
        txt.tag_configure('check',   font=('Consolas', 9),  foreground='#116611')
        txt.tag_configure('no',      font=('Consolas', 9),  foreground='#aa2222')

        txt.insert(tk.END, f'共振解析結果  ── {len(res)} 件検出\n', 'title')
        txt.insert(tk.END, '━' * 62 + '\n\n', 'sep')

        for i, r in enumerate(res, 1):
            stars = _STARS[min(r['score'], 5)]
            header = f'[{i}]  {r["freq"]:.2f} Hz   {r["ch"]}   {stars}\n'
            txt.insert(tk.END, header, 'head')
            for line in r['evidence']:
                tag = 'check' if '✓' in line else 'no' if '×' in line else 'ev'
                txt.insert(tk.END, f'     {line}\n', tag)
            txt.insert(tk.END, '\n')

        txt.configure(state=tk.DISABLED)

        # ボタン行
        bf = tk.Frame(win, bg='#ebebeb', pady=4)
        bf.pack(side=tk.BOTTOM, fill=tk.X)
        def _copy():
            content = txt.get('1.0', tk.END)
            win.clipboard_clear(); win.clipboard_append(content)
            btn_copy.configure(text='コピーしました')
            win.after(1500, lambda: btn_copy.configure(text='テキストをコピー'))
        btn_copy = tk.Button(bf, text='テキストをコピー', command=_copy,
                             font=('Arial', 9))
        btn_copy.pack(side=tk.LEFT, padx=8)
        tk.Button(bf, text='閉じる', command=win.destroy,
                  font=('Arial', 9)).pack(side=tk.RIGHT, padx=8)

    # ── 共振解析 実行 ────────────────────────────────────────────────────────
    def run_resonance_analysis():
        if not _SCIPY:
            messagebox.showerror(
                '依存ライブラリなし',
                'scipy がインストールされていません。\n'
                'ターミナルで  pip install scipy  を実行してください。')
            return
        btn_res.configure(text='解析中…', state=tk.DISABLED)
        root.update_idletasks()
        try:
            freq_range = ax.get_xlim()
            res = find_resonances(all_data, freq_range=freq_range)
        finally:
            btn_res.configure(text='共振解析', state=tk.NORMAL)
        if not res:
            messagebox.showinfo('共振解析', '有意なピークが検出されませんでした。')
            return
        resonance_results[0] = res
        _redraw()                  # アノテーション付きで再描画
        _show_result_window(res, freq_range)

    def clear_annotations():
        resonance_results[0] = None
        _redraw()

    btn_res.configure(command=run_resonance_analysis)
    btn_ann_clear.configure(command=clear_annotations)

    # ── Combobox ────────────────────────────────────────────────────────────
    def on_kind_change(event):
        _save_ranges()
        label = kind_var.get()
        for k in avail_kinds:
            if KIND_LABEL[k] == label:
                cur_kind[0] = k; break
        _load_ranges()
        _rebuild_ch_panel()
        _redraw()

    combo.bind('<<ComboboxSelected>>', on_kind_change)

    # ── スケール・Apply・Save ─────────────────────────────────────────────────
    def on_xlog(_): xscale[cur_kind[0]] = 'log';    _redraw()
    def on_xlin(_): xscale[cur_kind[0]] = 'linear'; _redraw()
    def on_ylog(_): yscale[cur_kind[0]] = 'log';    _redraw()
    def on_ylin(_): yscale[cur_kind[0]] = 'linear'; _redraw()

    btn_xlog.on_clicked(on_xlog)
    btn_xlin.on_clicked(on_xlin)
    btn_ylog.on_clicked(on_ylog)
    btn_ylin.on_clicked(on_ylin)
    btn_apply.on_clicked(lambda _: _redraw())

    def on_save(_):
        k   = cur_kind[0]
        out = os.path.join(os.path.dirname(paths[0]), f'VTplot_{k}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')
    btn_save.on_clicked(on_save)

    # ── 初期描画 ─────────────────────────────────────────────────────────────
    _rebuild_ch_panel()
    _redraw()
    root.mainloop()


if __name__ == '__main__':
    main()
