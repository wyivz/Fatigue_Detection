# -*- coding: utf-8 -*-
"""Hikrobot MVS camera helpers: enumerate / open / grab BGR frames.

Compatible with mono and color GigE (and USB3 Vision) cameras:
- Sensor-aware PixelFormat (Mono8 vs Bayer*8 vs RGB/BGR8)
- Primary grab via MV_CC_GetImageForBGR (SDK converts mono/Bayer/high-bit)
- Fallback raw grab + OpenCV / ConvertPixelType decode
"""
from __future__ import annotations

import logging
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("detection.camera")

_MVS_DIR = os.path.dirname(os.path.abspath(__file__))
_MVIMPORT_DIR = os.path.join(_MVS_DIR, "MvImport")
if _MVIMPORT_DIR not in sys.path:
    sys.path.insert(0, _MVIMPORT_DIR)

_sdk_error: Optional[str] = None
_sdk = None

# Pixel-type → preferred 8-bit host format (symbolic GenICam name)
_BAYER_TO_HOST8 = {
    # 8-bit
    17301512: "BayerGR8",  # GR8
    17301513: "BayerRG8",
    17301514: "BayerGB8",
    17301515: "BayerBG8",
    # 10/12/16 unpacked + packed → same pattern *8
    17825804: "BayerGR8",
    17825805: "BayerRG8",
    17825806: "BayerGB8",
    17825807: "BayerBG8",
    17825808: "BayerGR8",
    17825809: "BayerRG8",
    17825810: "BayerGB8",
    17825811: "BayerBG8",
    17563686: "BayerGR8",
    17563687: "BayerRG8",
    17563688: "BayerGB8",
    17563689: "BayerBG8",
    17563690: "BayerGR8",
    17563691: "BayerRG8",
    17563692: "BayerGB8",
    17563693: "BayerBG8",
    17825838: "BayerGR8",
    17825839: "BayerRG8",
    17825840: "BayerGB8",
    17825841: "BayerBG8",
}

_MONO_PIXELS = {
    17301505,  # Mono8
    17301506,  # Mono8 Signed
    17825795,  # Mono10
    17563652,  # Mono10 Packed
    17825797,  # Mono12
    17563654,  # Mono12 Packed
    17825829,  # Mono14
    17825799,  # Mono16
}

_RGB_PIXELS = {
    35127316,  # RGB8
    35127317,  # BGR8
    35651606,  # RGBA8
    35651607,  # BGRA8
}

_YUV_PIXELS = {
    34603039,  # YUV422_Packed (UYVY)
    34603058,  # YUV422_YUYV_Packed
}


def _load_sdk():
    global _sdk, _sdk_error
    if _sdk is not None:
        return _sdk
    try:
        if _MVIMPORT_DIR not in sys.path:
            sys.path.insert(0, _MVIMPORT_DIR)
        import MvCameraControl_class as mvs  # noqa: WPS433
        from CameraParams_const import (  # noqa: WPS433
            MV_ACCESS_Exclusive,
            MV_GIGE_DEVICE,
            MV_USB_DEVICE,
        )
        from CameraParams_header import (  # noqa: WPS433
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_CC_PIXEL_CONVERT_PARAM,
            MV_FRAME_OUT_INFO_EX,
            MVCC_ENUMVALUE,
            MVCC_FLOATVALUE,
            MVCC_INTVALUE,
        )
        from MvErrorDefine_const import MV_OK  # noqa: WPS433
        from PixelType_header import (  # noqa: WPS433
            PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_Mono8,
            PixelType_Gvsp_Mono16,
            PixelType_Gvsp_RGB8_Packed,
            PixelType_Gvsp_YUV422_Packed,
            PixelType_Gvsp_YUV422_YUYV_Packed,
        )

        if getattr(mvs, "MvCamCtrldll", None) is None:
            err = os.environ.get("MVS_DLL_LOAD_ERROR") or "MvCameraControl.dll not found"
            _sdk_error = (
                "MVS Runtime not found. Install Hikrobot MVS 4.6.3 (64-bit) "
                "or set MVS_RUNTIME_DIR / HIK_CAMERA_SDK_MVS_LIBRARY. Detail: %s" % err
            )
            return None

        class _Bundle:
            pass

        b = _Bundle()
        b.mvs = mvs
        b.MV_OK = MV_OK
        b.MV_GIGE_DEVICE = MV_GIGE_DEVICE
        b.MV_USB_DEVICE = MV_USB_DEVICE
        b.MV_ACCESS_Exclusive = MV_ACCESS_Exclusive
        b.MV_CC_DEVICE_INFO = MV_CC_DEVICE_INFO
        b.MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST
        b.MV_CC_PIXEL_CONVERT_PARAM = MV_CC_PIXEL_CONVERT_PARAM
        b.MV_FRAME_OUT_INFO_EX = MV_FRAME_OUT_INFO_EX
        b.MVCC_INTVALUE = MVCC_INTVALUE
        b.MVCC_FLOATVALUE = MVCC_FLOATVALUE
        b.MVCC_ENUMVALUE = MVCC_ENUMVALUE
        b.PixelType_Gvsp_BGR8_Packed = PixelType_Gvsp_BGR8_Packed
        b.PixelType_Gvsp_RGB8_Packed = PixelType_Gvsp_RGB8_Packed
        b.PixelType_Gvsp_Mono8 = PixelType_Gvsp_Mono8
        b.PixelType_Gvsp_Mono16 = PixelType_Gvsp_Mono16
        b.PixelType_Gvsp_BayerGR8 = PixelType_Gvsp_BayerGR8
        b.PixelType_Gvsp_BayerRG8 = PixelType_Gvsp_BayerRG8
        b.PixelType_Gvsp_BayerGB8 = PixelType_Gvsp_BayerGB8
        b.PixelType_Gvsp_BayerBG8 = PixelType_Gvsp_BayerBG8
        b.PixelType_Gvsp_YUV422_Packed = PixelType_Gvsp_YUV422_Packed
        b.PixelType_Gvsp_YUV422_YUYV_Packed = PixelType_Gvsp_YUV422_YUYV_Packed
        _sdk = b
        _sdk_error = None
        return _sdk
    except Exception as exc:  # noqa: BLE001
        _sdk_error = str(exc)
        return None


def is_mvs_available() -> Tuple[bool, Optional[str]]:
    sdk = _load_sdk()
    if sdk is None:
        return False, _sdk_error or "MVS SDK unavailable"
    return True, None


def _bytes_to_str(buf) -> str:
    try:
        return bytes(buf).split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return ""


def _ip_to_str(ip_uint: int) -> str:
    return "%d.%d.%d.%d" % (
        (ip_uint >> 24) & 0xFF,
        (ip_uint >> 16) & 0xFF,
        (ip_uint >> 8) & 0xFF,
        ip_uint & 0xFF,
    )


def classify_pixel_type(pixel: int) -> str:
    """Return 'mono' | 'bayer' | 'rgb' | 'yuv' | 'other' for a GVSP pixel type."""
    p = int(pixel)
    if p in _MONO_PIXELS:
        return "mono"
    if p in _BAYER_TO_HOST8:
        return "bayer"
    if p in _RGB_PIXELS:
        return "rgb"
    if p in _YUV_PIXELS:
        return "yuv"
    # HB_* codes (0x81xxxxxx): classify by low 16 bits (same ids as standard GVSP)
    low = p & 0xFFFF
    if 0x0001 <= low <= 0x0007:
        return "mono"
    if 0x0008 <= low <= 0x0013 or low == 0x0046:
        return "bayer"
    if low in (0x0014, 0x0015, 0x0016, 0x0017):
        return "rgb"
    if low in (0x001F, 0x0032):
        return "yuv"
    return "other"


def enumerate_devices() -> Dict[str, Any]:
    """Return GigE (and USB3 Vision) devices for UI selection."""
    ok, err = is_mvs_available()
    if not ok:
        return {"gige": [], "mvs_available": False, "mvs_error": err}

    sdk = _load_sdk()
    device_list = sdk.MV_CC_DEVICE_INFO_LIST()
    memset(byref(device_list), 0, sizeof(sdk.MV_CC_DEVICE_INFO_LIST))
    ret = sdk.mvs.MvCamera.MV_CC_EnumDevices(
        sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, device_list
    )
    if ret != sdk.MV_OK:
        return {
            "gige": [],
            "mvs_available": True,
            "mvs_error": "EnumDevices failed: 0x%x" % ret,
        }

    devices = []
    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        item = {
            "index": i,
            "model": "",
            "serial": "",
            "ip": "",
            "transport": "unknown",
            "display_name": "Device %d" % i,
        }
        if info.nTLayerType == sdk.MV_GIGE_DEVICE:
            g = info.SpecialInfo.stGigEInfo
            item["transport"] = "GigE"
            item["model"] = _bytes_to_str(g.chModelName)
            item["serial"] = _bytes_to_str(g.chSerialNumber)
            item["ip"] = _ip_to_str(g.nCurrentIp)
            name = item["model"] or "GigE"
            item["display_name"] = "%s [%s] #%d" % (name, item["ip"] or item["serial"], i)
        elif info.nTLayerType == sdk.MV_USB_DEVICE:
            u = info.SpecialInfo.stUsb3VInfo
            item["transport"] = "USB3"
            item["model"] = _bytes_to_str(u.chModelName)
            item["serial"] = _bytes_to_str(u.chSerialNumber)
            name = item["model"] or "USB3"
            item["display_name"] = "%s [%s] #%d" % (name, item["serial"] or "usb", i)
        else:
            item["display_name"] = "Device #%d (type=0x%x)" % (i, info.nTLayerType)
        devices.append(item)

    return {"gige": devices, "mvs_available": True, "mvs_error": None}


class HikCamera:
    """Open one MVS device and grab BGR numpy frames (mono or color)."""

    def __init__(self):
        self._cam = None
        self._buf = None
        self._payload = 0
        self._bgr_buf = None
        self._bgr_capacity = 0
        self._opened = False
        self._grabbing = False
        self._sdk = None
        self._sensor_family = "other"  # mono | bayer | rgb | yuv | other
        self._prefer_sdk_bgr = True
        self.last_diag: Dict[str, Any] = {}
        self._dark_streak = 0
        self._fixed_exposure_locked = False
        # Diagnostics: how many times recover_stream() has kicked in this session.
        self._reconnect_count = 0

    def open_by_index(self, index: int = 0) -> None:
        sdk = _load_sdk()
        if sdk is None:
            raise RuntimeError(_sdk_error or "MVS unavailable")

        device_list = sdk.MV_CC_DEVICE_INFO_LIST()
        memset(byref(device_list), 0, sizeof(sdk.MV_CC_DEVICE_INFO_LIST))
        ret = sdk.mvs.MvCamera.MV_CC_EnumDevices(
            sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, device_list
        )
        if ret != sdk.MV_OK:
            raise RuntimeError("EnumDevices failed: 0x%x" % ret)
        if index < 0 or index >= device_list.nDeviceNum:
            raise RuntimeError("Device index %s out of range (n=%s)" % (index, device_list.nDeviceNum))

        info = cast(device_list.pDeviceInfo[index], POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        self._open_info(sdk, info)

    def open_by_ip(self, ip: str) -> None:
        data = enumerate_devices()
        if not data.get("mvs_available"):
            raise RuntimeError(data.get("mvs_error") or "MVS unavailable")
        for d in data.get("gige") or []:
            if d.get("ip") == ip:
                self.open_by_index(int(d["index"]))
                return
        raise RuntimeError("No GigE camera with IP %s" % ip)

    def _open_info(self, sdk, info) -> None:
        cam = sdk.mvs.MvCamera()
        ret = cam.MV_CC_CreateHandle(info)
        if ret != sdk.MV_OK:
            raise RuntimeError("CreateHandle failed: 0x%x" % ret)

        access_modes = [
            ("Exclusive", getattr(sdk, "MV_ACCESS_Exclusive", 1)),
            ("ExclusiveWithSwitch", getattr(sdk, "MV_ACCESS_ExclusiveWithSwitch", 2)),
            ("Control", getattr(sdk, "MV_ACCESS_Control", 3)),
        ]
        try:
            from CameraParams_const import (  # noqa: WPS433
                MV_ACCESS_Control,
                MV_ACCESS_Exclusive,
                MV_ACCESS_ExclusiveWithSwitch,
            )
            access_modes = [
                ("Exclusive", MV_ACCESS_Exclusive),
                ("ExclusiveWithSwitch", MV_ACCESS_ExclusiveWithSwitch),
                ("Control", MV_ACCESS_Control),
            ]
        except Exception:  # noqa: BLE001
            pass

        last_ret = None
        opened = False
        for _name, mode in access_modes:
            ret = cam.MV_CC_OpenDevice(mode, 0)
            last_ret = ret
            if ret == sdk.MV_OK:
                opened = True
                break
        if not opened:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                "OpenDevice failed: 0x%x (close MVS client preview / free camera, then retry)"
                % (last_ret or 0)
            )

        is_gige = bool(getattr(info, "nTLayerType", 0) == sdk.MV_GIGE_DEVICE)
        self._configure_stream(cam, sdk, is_gige=is_gige)
        self._apply_default_auto_exposure_gain(cam, sdk)

        st_param = sdk.MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(sdk.MVCC_INTVALUE))
        ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != sdk.MV_OK or st_param.nCurValue <= 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise RuntimeError("Get PayloadSize failed: 0x%x" % ret)

        self._payload = int(st_param.nCurValue)
        # Extra headroom for packed / HB payloads
        self._buf = (c_ubyte * max(self._payload, self._payload + 64))()
        self._cam = cam
        self._sdk = sdk
        self._opened = True
        self._alloc_bgr_buf_from_roi(cam, sdk)
        self.last_diag.update(
            {
                "payload": self._payload,
                "is_gige": is_gige,
                "sensor_family": self._sensor_family,
                "is_mono": self._sensor_family == "mono",
            }
        )

    def _get_int(self, cam, sdk, key: str) -> Optional[int]:
        st = sdk.MVCC_INTVALUE()
        memset(byref(st), 0, sizeof(sdk.MVCC_INTVALUE))
        try:
            ret = cam.MV_CC_GetIntValue(key, st)
            if ret == sdk.MV_OK and st.nCurValue > 0:
                return int(st.nCurValue)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _get_enum_cur(self, cam, sdk, key: str) -> Optional[int]:
        st = sdk.MVCC_ENUMVALUE()
        memset(byref(st), 0, sizeof(sdk.MVCC_ENUMVALUE))
        try:
            ret = cam.MV_CC_GetEnumValue(key, st)
            if ret == sdk.MV_OK:
                return int(st.nCurValue)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _alloc_bgr_buf_from_roi(self, cam, sdk) -> None:
        w = self._get_int(cam, sdk, "Width") or 0
        h = self._get_int(cam, sdk, "Height") or 0
        if w > 0 and h > 0:
            self._ensure_bgr_buf(w, h)
            self.last_diag["roi"] = "%sx%s" % (w, h)

    def _ensure_bgr_buf(self, w: int, h: int) -> None:
        need = int(w) * int(h) * 3
        if need <= 0:
            return
        if self._bgr_capacity < need:
            self._bgr_buf = (c_ubyte * need)()
            self._bgr_capacity = need

    @staticmethod
    def _align_int(value: int, inc: int, lo: int, hi: int) -> int:
        """Align value down to GenICam increment and clamp to [lo, hi]."""
        inc = max(1, int(inc or 1))
        lo = int(lo)
        hi = int(hi)
        v = int(value)
        v = v - (v % inc)
        if v < lo:
            v = lo + ((inc - (lo % inc)) % inc)
        if v > hi:
            v = hi - (hi % inc)
        return max(lo, min(hi, v))

    def _apply_stream_roi(
        self,
        cam,
        sdk,
        max_width: int = 1920,
        max_height: int = 1600,
    ) -> None:
        """
        Cap GigE ROI for reliable realtime streaming while keeping a wide FOV.

        Full 5MP (e.g. 2448x2048) can yield MV_E_NODATA on ordinary 1GbE.
        Use a large centered ROI (default 1920x1600) so faces stay in frame;
        detection still downscales independently.
        """
        wr = self._get_int_range(cam, sdk, "Width")
        hr = self._get_int_range(cam, sdk, "Height")
        if wr is None or hr is None:
            return
        _cw, wmin, wmax, winc = wr
        _ch, hmin, hmax, hinc = hr
        if wmax <= 0 or hmax <= 0:
            return

        target_w = min(int(max_width), int(wmax))
        target_h = min(int(max_height), int(hmax))
        # Prefer sensor aspect so the crop is a true window, not a stretched box
        aspect_h = int(round(target_w * (float(hmax) / float(wmax))))
        if aspect_h <= int(hmax):
            target_h = min(target_h, aspect_h)

        tw = self._align_int(target_w, winc, wmin, wmax)
        th = self._align_int(target_h, hinc, hmin, hmax)

        # Match target size (expand small leftover ROIs; shrink oversized full-sensor)
        same = abs(int(_cw) - tw) <= int(winc) and abs(int(_ch) - th) <= int(hinc)
        if same:
            # Still re-center if offset drifted
            oxr = self._get_int_range(cam, sdk, "OffsetX")
            oyr = self._get_int_range(cam, sdk, "OffsetY")
            if oxr is not None and oyr is not None:
                ox = self._align_int(
                    max(0, (int(wmax) - tw) // 2),
                    int(oxr[3] or 1),
                    int(oxr[1]),
                    max(int(oxr[1]), int(oxr[2])),
                )
                oy = self._align_int(
                    max(0, (int(hmax) - th) // 2),
                    int(oyr[3] or 1),
                    int(oyr[1]),
                    max(int(oyr[1]), int(oyr[2])),
                )
                try:
                    cam.MV_CC_SetIntValue("OffsetX", ox)
                    cam.MV_CC_SetIntValue("OffsetY", oy)
                except Exception:  # noqa: BLE001
                    pass
            self.last_diag["stream_roi"] = "%sx%s (target)" % (tw, th)
            return

        # Offsets must be zero before changing Width/Height on many Hikrobot models
        try:
            cam.MV_CC_SetIntValue("OffsetX", 0)
            cam.MV_CC_SetIntValue("OffsetY", 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            cam.MV_CC_SetIntValue("Width", tw)
            cam.MV_CC_SetIntValue("Height", th)
        except Exception as exc:  # noqa: BLE001
            self.last_diag["stream_roi_error"] = str(exc)
            return

        gw = self._get_int(cam, sdk, "Width") or tw
        gh = self._get_int(cam, sdk, "Height") or th
        oxr = self._get_int_range(cam, sdk, "OffsetX")
        oyr = self._get_int_range(cam, sdk, "OffsetY")
        if oxr is not None:
            _ox, oxmin, oxmax, oxinc = oxr
            ox = self._align_int(
                max(0, (int(wmax) - int(gw)) // 2),
                int(oxinc or 1),
                int(oxmin),
                max(int(oxmin), int(oxmax)),
            )
            try:
                cam.MV_CC_SetIntValue("OffsetX", ox)
            except Exception:  # noqa: BLE001
                pass
        if oyr is not None:
            _oy, oymin, oymax, oyinc = oyr
            oy = self._align_int(
                max(0, (int(hmax) - int(gh)) // 2),
                int(oyinc or 1),
                int(oymin),
                max(int(oymin), int(oymax)),
            )
            try:
                cam.MV_CC_SetIntValue("OffsetY", oy)
            except Exception:  # noqa: BLE001
                pass

        gw = self._get_int(cam, sdk, "Width") or gw
        gh = self._get_int(cam, sdk, "Height") or gh
        ox = self._get_int(cam, sdk, "OffsetX") or 0
        oy = self._get_int(cam, sdk, "OffsetY") or 0
        self.last_diag["stream_roi"] = "%sx%s@%s,%s (cap %sx%s)" % (
            gw,
            gh,
            ox,
            oy,
            tw,
            th,
        )

    def _configure_stream(self, cam, sdk, is_gige: bool) -> None:
        """Continuous free-run + GigE packet size; sensor-aware pixel format."""
        try:
            cam.MV_CC_SetEnumValue("TriggerMode", 0)
        except Exception:  # noqa: BLE001
            pass
        self._set_enum(cam, sdk, "AcquisitionMode", "Continuous", 2)

        if is_gige:
            try:
                pkt = int(cam.MV_CC_GetOptimalPacketSize())
                if 500 <= pkt <= 16384:
                    try:
                        cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)
                    except Exception:  # noqa: BLE001
                        pass
                    self.last_diag["packet_size"] = pkt
            except Exception:  # noqa: BLE001
                pass

            # Inter-packet delay: SCPD=0 maxes throughput but often drops frames
            # on ordinary 1GbE NICs with large ROI payloads.
            scpd = 2000
            try:
                from detection.utils.config_cache import get_configs

                raw = (get_configs() or {}).get("mvs_gev_scpd")
                if raw is not None and str(raw).strip() != "":
                    scpd = max(0, int(float(raw)))
            except Exception:  # noqa: BLE001
                pass
            try:
                ret = cam.MV_CC_SetIntValue("GevSCPD", int(scpd))
                if ret == sdk.MV_OK:
                    self.last_diag["gev_scpd"] = int(scpd)
            except Exception:  # noqa: BLE001
                pass

            # Wide FOV: mvs_stream_max_width=0 → full sensor (no ROI crop)
            stream_w, stream_h = 0, 0
            try:
                from detection.utils.config_cache import get_configs

                cfg = get_configs() or {}
                if cfg.get("mvs_stream_max_width") not in (None, ""):
                    stream_w = int(float(cfg.get("mvs_stream_max_width")))
                if cfg.get("mvs_stream_max_height") not in (None, ""):
                    stream_h = int(float(cfg.get("mvs_stream_max_height")))
            except Exception:  # noqa: BLE001
                pass
            if stream_w > 0 and stream_h > 0:
                self._apply_stream_roi(
                    cam, sdk, max_width=max(960, stream_w), max_height=max(720, stream_h)
                )
            else:
                self.last_diag["stream_roi"] = "full_sensor"

            # Cap FPS so GigE stays within link budget after ROI/SCPD
            try:
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            except Exception:  # noqa: BLE001
                pass
            try:
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 15.0)
                self.last_diag["acq_fps"] = 15.0
            except Exception:  # noqa: BLE001
                pass

        self._prefer_host_pixel_format(cam, sdk)

        # Color-only: continuous white balance (ignore failures on mono)
        if self._sensor_family in ("bayer", "rgb", "yuv"):
            self._set_enum(cam, sdk, "BalanceWhiteAuto", "Continuous", 1)
            try:
                cam.MV_CC_SetBayerCvtQuality(1)
            except Exception:  # noqa: BLE001
                pass

    def _list_supported_pixel_values(self, cam, sdk) -> List[int]:
        st = sdk.MVCC_ENUMVALUE()
        memset(byref(st), 0, sizeof(sdk.MVCC_ENUMVALUE))
        try:
            ret = cam.MV_CC_GetEnumValue("PixelFormat", st)
            if ret != sdk.MV_OK:
                return []
            n = int(st.nSupportedNum or 0)
            n = max(0, min(n, len(st.nSupportValue)))
            return [int(st.nSupportValue[i]) for i in range(n)]
        except Exception:  # noqa: BLE001
            return []

    def _resolve_sensor_family(self, cam, sdk, cur: Optional[int]) -> str:
        """
        Decide mono vs color from current + supported PixelFormat list.

        Color cameras often also expose Mono8 — if any Bayer/RGB/YUV is
        supported, treat as color so we do not silently switch to mono.
        """
        supported = self._list_supported_pixel_values(cam, sdk)
        families = {classify_pixel_type(v) for v in supported} if supported else set()
        if "bayer" in families or "rgb" in families or "yuv" in families:
            if cur is not None and classify_pixel_type(cur) in ("bayer", "rgb", "yuv"):
                return classify_pixel_type(cur)
            if "bayer" in families:
                return "bayer"
            if "rgb" in families:
                return "rgb"
            return "yuv"
        if "mono" in families:
            return "mono"
        if cur is not None:
            return classify_pixel_type(cur)
        return "other"

    def _prefer_host_pixel_format(self, cam, sdk) -> None:
        """
        Choose a host-friendly 8-bit format without crossing mono↔color.

        Mono → Mono8 only. Color Bayer → same-pattern Bayer*8.
        Never force Bayer onto mono-only sensors; never force Mono onto color.
        """
        cur = self._get_enum_cur(cam, sdk, "PixelFormat")
        family = self._resolve_sensor_family(cam, sdk, cur)
        self._sensor_family = family
        self.last_diag["pixel_type_open"] = ("0x%x" % cur) if cur is not None else None
        self.last_diag["sensor_family"] = family

        keep_ok = {
            int(sdk.PixelType_Gvsp_Mono8),
            int(sdk.PixelType_Gvsp_BayerRG8),
            int(sdk.PixelType_Gvsp_BayerGR8),
            int(sdk.PixelType_Gvsp_BayerGB8),
            int(sdk.PixelType_Gvsp_BayerBG8),
            int(sdk.PixelType_Gvsp_BGR8_Packed),
            int(sdk.PixelType_Gvsp_RGB8_Packed),
        }
        # Keep only if current 8-bit matches the resolved family
        if cur in keep_ok:
            cur_fam = classify_pixel_type(cur)
            if (family == "mono" and cur_fam == "mono") or (
                family != "mono" and cur_fam != "mono"
            ):
                self.last_diag["pixel_format_set"] = "keep"
                self.last_diag["is_mono"] = family == "mono"
                return

        candidates: List[str] = []
        if family == "mono":
            candidates = ["Mono8"]
        elif family == "bayer":
            preferred = _BAYER_TO_HOST8.get(int(cur or 0))
            if preferred:
                candidates.append(preferred)
            for fmt in ("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"):
                if fmt not in candidates:
                    candidates.append(fmt)
        elif family == "rgb":
            candidates = ["BGR8Packed", "RGB8Packed"]
        elif family == "yuv":
            candidates = ["BGR8Packed", "RGB8Packed"]  # optional; YUV kept if set fails
        else:
            # Truly unknown: prefer color formats first, then mono (safer for mixed lists)
            candidates = [
                "BayerRG8",
                "BayerGB8",
                "BayerGR8",
                "BayerBG8",
                "BGR8Packed",
                "RGB8Packed",
                "Mono8",
            ]

        for fmt in candidates:
            if self._set_enum(cam, sdk, "PixelFormat", fmt, -1):
                self.last_diag["pixel_format_set"] = fmt
                break

        after = self._get_enum_cur(cam, sdk, "PixelFormat")
        if after is not None:
            self.last_diag["pixel_type_after"] = "0x%x" % after
            # Keep resolved family (supported-list based), not just current value
            if family == "other":
                self._sensor_family = classify_pixel_type(after)
            else:
                self._sensor_family = family
            self.last_diag["sensor_family"] = self._sensor_family
            self.last_diag["is_mono"] = self._sensor_family == "mono"

    @staticmethod
    def _set_enum(cam, sdk, key: str, symbolic: str, numeric: int) -> bool:
        """Try GenICam enum by name, then by numeric value. Returns True on success."""
        try:
            ret = cam.MV_CC_SetEnumValueByString(key, symbolic)
            if ret == sdk.MV_OK:
                return True
        except Exception:  # noqa: BLE001
            pass
        if numeric >= 0:
            try:
                ret = cam.MV_CC_SetEnumValue(key, numeric)
                if ret == sdk.MV_OK:
                    return True
            except Exception:  # noqa: BLE001
                pass
        return False

    def _apply_default_auto_exposure_gain(self, cam, sdk) -> None:
        """Try Continuous AE/AG; if unsupported, leave manual Off for later software calibration."""
        continuous = 2
        ok_exp = self._set_enum(cam, sdk, "ExposureAuto", "Continuous", continuous)
        ok_gain = self._set_enum(cam, sdk, "GainAuto", "Continuous", continuous)
        if not ok_gain:
            ok_gain = self._set_enum(cam, sdk, "Gain", "Continuous", continuous)

        if ok_exp:
            for key, val in (
                ("AutoExposureTimeUpperLimit", 200000.0),
                ("AutoExposureTimeUpperLimit", 200000),
                ("ExposureAutoUpperLimit", 200000.0),
            ):
                try:
                    if isinstance(val, float):
                        cam.MV_CC_SetFloatValue(key, float(val))
                    else:
                        cam.MV_CC_SetIntValue(key, int(val))
                    self.last_diag["ae_upper"] = val
                    break
                except Exception:  # noqa: BLE001
                    continue
            for key, val in (("AutoExposureTimeLowerLimit", 100.0), ("ExposureAutoLowerLimit", 100)):
                try:
                    if isinstance(val, float):
                        cam.MV_CC_SetFloatValue(key, float(val))
                    else:
                        cam.MV_CC_SetIntValue(key, int(val))
                    break
                except Exception:  # noqa: BLE001
                    continue
        else:
            # Hardware AE missing (e.g. JHEM506GC): seed a mid exposure for calibration
            self._set_enum(cam, sdk, "ExposureAuto", "Off", 0)
            self._set_enum(cam, sdk, "GainAuto", "Off", 0)
            self._set_exposure_gain(cam, sdk, exposure_us=12000.0, gain_db=4.0)

        self.last_diag["ae"] = bool(ok_exp)
        self.last_diag["ag"] = bool(ok_gain)

    def _get_float_range(self, cam, sdk, key: str) -> Optional[Tuple[float, float, float]]:
        st = sdk.MVCC_FLOATVALUE()
        memset(byref(st), 0, sizeof(sdk.MVCC_FLOATVALUE))
        try:
            ret = cam.MV_CC_GetFloatValue(key, st)
            if ret == sdk.MV_OK:
                return float(st.fCurValue), float(st.fMin), float(st.fMax)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _get_int_range(
        self, cam, sdk, key: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return (cur, min, max, inc) for an integer GenICam node."""
        st = sdk.MVCC_INTVALUE()
        memset(byref(st), 0, sizeof(sdk.MVCC_INTVALUE))
        try:
            ret = cam.MV_CC_GetIntValue(key, st)
            if ret == sdk.MV_OK:
                return (
                    float(st.nCurValue),
                    float(st.nMin),
                    float(st.nMax),
                    float(st.nInc or 1),
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    def _set_exposure_gain(
        self, cam, sdk, exposure_us: float, gain_db: float
    ) -> Tuple[bool, bool]:
        ok_exp = False
        ok_gain = False
        for setter, key, val in (
            (cam.MV_CC_SetFloatValue, "ExposureTime", float(exposure_us)),
            (cam.MV_CC_SetIntValue, "ExposureTime", int(round(exposure_us))),
        ):
            try:
                ret = setter(key, val)
                if ret == sdk.MV_OK:
                    ok_exp = True
                    break
            except Exception:  # noqa: BLE001
                continue
        for setter, key, val in (
            (cam.MV_CC_SetFloatValue, "Gain", float(gain_db)),
            (cam.MV_CC_SetFloatValue, "GainRaw", float(gain_db)),
        ):
            try:
                ret = setter(key, val)
                if ret == sdk.MV_OK:
                    ok_gain = True
                    break
            except Exception:  # noqa: BLE001
                continue
        return ok_exp, ok_gain

    def _force_manual_exposure(self, exposure_us: float = 15000.0, gain_db: float = 8.0) -> None:
        """Rescue path when frames stay near-black (skipped once fixed exposure is locked)."""
        if self._fixed_exposure_locked:
            return
        cam, sdk = self._cam, self._sdk
        if cam is None or sdk is None:
            return
        self._set_enum(cam, sdk, "ExposureAuto", "Off", 0)
        self._set_enum(cam, sdk, "GainAuto", "Off", 0)
        self._set_exposure_gain(cam, sdk, exposure_us=exposure_us, gain_db=gain_db)
        self.last_diag["manual_rescue"] = {
            "exposure_us": exposure_us,
            "gain_db": gain_db,
        }

    def start_grab(self) -> None:
        if not self._opened:
            raise RuntimeError("Camera not opened")
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != self._sdk.MV_OK:
            raise RuntimeError("StartGrabbing failed: 0x%x" % ret)
        self._grabbing = True
        self._prefer_sdk_bgr = True
        try:
            self._cam.MV_CC_ClearImageBuffer()
        except Exception:  # noqa: BLE001
            pass

    def recover_stream(self, tighten_roi: bool = True) -> Dict[str, Any]:
        """
        Recover from sustained MV_E_NODATA: bump GevSCPD, optionally shrink ROI,
        restart grabbing. Safe to call from the grab loop.
        """
        cam, sdk = self._cam, self._sdk
        if cam is None or sdk is None or not self._opened:
            return dict(self.last_diag)

        self._reconnect_count += 1
        self.last_diag["reconnect_count"] = self._reconnect_count
        logger.warning(
            "GigE stream recovery #%d triggered (tighten_roi=%s)",
            self._reconnect_count, tighten_roi,
        )

        steps = []
        try:
            if self._grabbing:
                cam.MV_CC_StopGrabbing()
                self._grabbing = False
                steps.append("stop")
        except Exception as exc:  # noqa: BLE001
            steps.append("stop_err:%s" % exc)

        # Increase inter-packet delay
        scpd_cur = 1500
        try:
            rng = self._get_int_range(cam, sdk, "GevSCPD")
            if rng is not None:
                scpd_cur = int(rng[0])
        except Exception:  # noqa: BLE001
            pass
        scpd_next = min(8000, max(1500, int(scpd_cur) + 1000))
        try:
            if cam.MV_CC_SetIntValue("GevSCPD", scpd_next) == sdk.MV_OK:
                self.last_diag["gev_scpd"] = scpd_next
                steps.append("scpd=%s" % scpd_next)
        except Exception:  # noqa: BLE001
            pass

        if tighten_roi:
            wr = self._get_int_range(cam, sdk, "Width")
            hr = self._get_int_range(cam, sdk, "Height")
            cur_w = int(wr[0]) if wr else 1920
            cur_h = int(hr[0]) if hr else 1600
            # Never shrink below a usable FOV for face tracking
            self._apply_stream_roi(
                cam,
                sdk,
                max_width=max(1280, cur_w * 5 // 6),
                max_height=max(960, cur_h * 5 // 6),
            )
            steps.append("roi=%s" % self.last_diag.get("stream_roi"))
            # Payload may change with ROI — refresh grab buffers
            try:
                st_param = sdk.MVCC_INTVALUE()
                memset(byref(st_param), 0, sizeof(sdk.MVCC_INTVALUE))
                ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
                if ret == sdk.MV_OK and st_param.nCurValue > 0:
                    self._payload = int(st_param.nCurValue)
                    self._buf = (c_ubyte * max(self._payload, self._payload + 64))()
                self._alloc_bgr_buf_from_roi(cam, sdk)
                steps.append("payload=%s" % self._payload)
            except Exception as exc:  # noqa: BLE001
                steps.append("payload_err:%s" % exc)

        try:
            ret = cam.MV_CC_StartGrabbing()
            if ret != sdk.MV_OK:
                raise RuntimeError("StartGrabbing failed: 0x%x" % ret)
            self._grabbing = True
            try:
                cam.MV_CC_ClearImageBuffer()
            except Exception:  # noqa: BLE001
                pass
            steps.append("start")
        except Exception as exc:  # noqa: BLE001
            steps.append("start_err:%s" % exc)

        self.last_diag["stream_recover"] = steps
        logger.info("GigE stream recovery #%d steps: %s", self._reconnect_count, steps)
        return dict(self.last_diag)

    def settle_exposure(self, frames: int = 25, timeout_ms: int = 500) -> Dict[str, Any]:
        """Backward-compatible: prefer hardware AE settle, else software fallback."""
        budget = max(1.0, float(frames) * float(timeout_ms) / 1000.0 * 0.35)
        if self.last_diag.get("ae"):
            return self.settle_hardware_ae(budget_sec=budget, timeout_ms=timeout_ms)
        return self.calibrate_fixed_exposure(budget_sec=budget, timeout_ms=timeout_ms)

    def settle_hardware_ae(
        self,
        budget_sec: float = 2.5,
        target_mean: float = 90.0,
        timeout_ms: int = 350,
        on_frame=None,
    ) -> Dict[str, Any]:
        """
        Let Continuous ExposureAuto/GainAuto climb; do not force Off.

        Returns needs_software_fallback=True when brightness stays unusable.
        """
        if not self._grabbing:
            self.last_diag["calib_error"] = "camera not grabbing"
            return dict(self.last_diag)

        deadline = time.time() + max(0.3, float(budget_sec))
        target = float(target_mean)
        means: List[float] = []
        last = None
        while time.time() < deadline:
            try:
                last = self.get_bgr_frame(timeout_ms=timeout_ms)
                m = float(np.mean(last))
                means.append(m)
                if on_frame is not None:
                    try:
                        on_frame(last)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001
                self.last_diag["settle_error"] = str(exc)
                break
            # Early stop once brightness is usable and stable
            if len(means) >= 3 and float(np.mean(means[-3:])) >= target * 0.85:
                break

        mean_last = float(means[-1]) if means else 0.0
        usable = mean_last >= max(20.0, target * 0.45)
        still_dark = mean_last < 15.0
        self.last_diag.update(
            {
                "exposure_mode": "hardware_ae",
                "fixed_exposure": False,
                "ae_settle_mean": round(mean_last, 2),
                "settle_mean": round(mean_last, 2),
                "ae_settled": bool(usable),
                "needs_software_fallback": bool(still_dark or not usable),
                "ae_settle_frames": len(means),
            }
        )
        return dict(self.last_diag)

    def calibrate_fixed_exposure(
        self,
        budget_sec: float = 4.0,
        target_mean: float = 105.0,
        timeout_ms: int = 400,
        on_frame=None,
    ) -> Dict[str, Any]:
        """
        Fallback: read frames and lock ExposureTime/Gain to a fixed suitable value.

        Used only when Continuous ExposureAuto/GainAuto is unsupported or
        hardware AE failed to reach usable brightness in time.
        """
        cam, sdk = self._cam, self._sdk
        if cam is None or sdk is None or not self._grabbing:
            self.last_diag["calib_error"] = "camera not grabbing"
            return dict(self.last_diag)

        deadline = time.time() + max(0.5, float(budget_sec))
        target = float(target_mean)
        lo = max(40.0, target - 25.0)
        hi = min(200.0, target + 25.0)

        self._set_enum(cam, sdk, "ExposureAuto", "Off", 0)
        self._set_enum(cam, sdk, "GainAuto", "Off", 0)

        exp_range = self._get_float_range(cam, sdk, "ExposureTime")
        if exp_range is None:
            exp_range = self._get_int_range(cam, sdk, "ExposureTime")
        gain_range = self._get_float_range(cam, sdk, "Gain")
        if gain_range is None:
            gain_range = self._get_float_range(cam, sdk, "GainRaw")

        # Sensible caps so FPS stays usable for realtime detection
        exp_min = 100.0
        exp_max = 80000.0
        exp_cur = 12000.0
        if exp_range is not None:
            exp_cur, exp_min, exp_max = exp_range[:3]
            exp_min = max(50.0, float(exp_min))
            exp_max = min(100000.0, max(exp_min + 1.0, float(exp_max)))
            exp_cur = float(np.clip(exp_cur, exp_min, exp_max))

        gain_min = 0.0
        gain_max = 16.0
        gain_cur = 4.0
        if gain_range is not None:
            gain_cur, gain_min, gain_max = gain_range
            gain_min = max(0.0, float(gain_min))
            gain_max = min(24.0, max(gain_min, float(gain_max)))
            gain_cur = float(np.clip(gain_cur, gain_min, gain_max))

        history: List[Dict[str, float]] = []
        mean_last = 0.0
        steps = 0

        def _measure() -> float:
            nonlocal mean_last
            last = None
            means_local: List[float] = []
            for _ in range(2):
                if time.time() >= deadline:
                    break
                try:
                    last = self.get_bgr_frame(timeout_ms=timeout_ms)
                    m = float(np.mean(last))
                    means_local.append(m)
                    if on_frame is not None and last is not None:
                        try:
                            on_frame(last)
                        except Exception:  # noqa: BLE001
                            pass
                except Exception:  # noqa: BLE001
                    break
            if means_local:
                mean_last = float(np.median(means_local))
            return mean_last

        # Initial apply + measure
        self._set_exposure_gain(cam, sdk, exp_cur, gain_cur)
        _measure()
        history.append({"exp_us": exp_cur, "gain_db": gain_cur, "mean": mean_last})

        while time.time() < deadline and steps < 14:
            if lo <= mean_last <= hi:
                break
            steps += 1
            if mean_last < lo:
                # Too dark: prefer longer exposure, then gain
                if exp_cur < exp_max * 0.98:
                    # Multiplicative step scaled by darkness
                    ratio = max(1.25, min(2.2, (target / max(mean_last, 1.0)) ** 0.7))
                    exp_cur = float(min(exp_max, exp_cur * ratio))
                elif gain_cur < gain_max - 0.05:
                    gain_cur = float(min(gain_max, gain_cur + max(1.0, (gain_max - gain_min) * 0.15)))
                else:
                    break
            else:
                # Too bright: reduce gain first, then exposure
                if gain_cur > gain_min + 0.05:
                    gain_cur = float(max(gain_min, gain_cur - max(1.0, (gain_max - gain_min) * 0.15)))
                elif exp_cur > exp_min * 1.02:
                    ratio = max(1.25, min(2.2, (mean_last / max(target, 1.0)) ** 0.7))
                    exp_cur = float(max(exp_min, exp_cur / ratio))
                else:
                    break
            self._set_exposure_gain(cam, sdk, exp_cur, gain_cur)
            _measure()
            history.append({"exp_us": exp_cur, "gain_db": gain_cur, "mean": mean_last})

        # Final lock (re-apply once; ignore further dark-streak rescue)
        self._set_enum(cam, sdk, "ExposureAuto", "Off", 0)
        self._set_enum(cam, sdk, "GainAuto", "Off", 0)
        self._set_exposure_gain(cam, sdk, exp_cur, gain_cur)
        if time.time() < deadline:
            _measure()
        self._fixed_exposure_locked = True
        self._dark_streak = 0

        self.last_diag.update(
            {
                "exposure_mode": "software_fallback",
                "fixed_exposure": True,
                "needs_software_fallback": False,
                "exposure_us": round(float(exp_cur), 1),
                "gain_db": round(float(gain_cur), 2),
                "calib_mean": round(float(mean_last), 2),
                "calib_target": target,
                "calib_steps": steps,
                "calib_ok": bool(lo <= mean_last <= hi),
                "calib_history": history[-8:],
                "settle_mean": round(float(mean_last), 2),
            }
        )
        return dict(self.last_diag)

    def get_bgr_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        if not self._grabbing:
            raise RuntimeError("Not grabbing")
        sdk = self._sdk
        cam = self._cam

        # --- Preferred path: SDK converts Mono / Bayer / high-bit → BGR8 ---
        if self._prefer_sdk_bgr and self._bgr_buf is not None and self._bgr_capacity > 0:
            frame_info = sdk.MV_FRAME_OUT_INFO_EX()
            memset(byref(frame_info), 0, sizeof(sdk.MV_FRAME_OUT_INFO_EX))
            ret = cam.MV_CC_GetImageForBGR(
                self._bgr_buf, self._bgr_capacity, frame_info, int(timeout_ms)
            )
            if ret == sdk.MV_OK:
                w = int(frame_info.nWidth or frame_info.nExtendWidth)
                h = int(frame_info.nHeight or frame_info.nExtendHeight)
                nlen = int(frame_info.nFrameLen or 0)
                if w > 0 and h > 0:
                    need = w * h * 3
                    if nlen >= need or self._bgr_capacity >= need:
                        out = np.frombuffer(
                            self._bgr_buf, dtype=np.uint8, count=need
                        ).reshape((h, w, 3)).copy()
                        self._update_diag(w, h, nlen or need, int(frame_info.enPixelType), out, path="GetImageForBGR")
                        return out
                    # Buffer too small (ROI changed) — grow and fall through to raw once
                    self._ensure_bgr_buf(w, h)
            elif ret == 0x8000000A:  # MV_E_NOENOUGH_BUF
                w = self._get_int(cam, sdk, "Width") or 0
                h = self._get_int(cam, sdk, "Height") or 0
                if w > 0 and h > 0:
                    self._ensure_bgr_buf(w, h)
            elif ret in (0x80000007, 0x8000000D, 0x8000000B):
                # NODATA / NOOUTBUF / ABNORMAL — transient; try raw path this frame
                pass
            elif ret == 0x80000001:  # MV_E_SUPPORT — camera/SDK cannot convert this way
                self._prefer_sdk_bgr = False
                self.last_diag["sdk_bgr_fallback"] = "0x%x" % ret
            # Other errors: still try raw decode once without permanently disabling

        # --- Fallback: raw payload + local decode ---
        frame_info = sdk.MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(sdk.MV_FRAME_OUT_INFO_EX))
        ret = cam.MV_CC_GetOneFrameTimeout(self._buf, self._payload, frame_info, timeout_ms)
        if ret != sdk.MV_OK:
            raise RuntimeError("GetOneFrameTimeout failed: 0x%x" % ret)

        w = int(frame_info.nWidth or frame_info.nExtendWidth)
        h = int(frame_info.nHeight or frame_info.nExtendHeight)
        nlen = int(frame_info.nFrameLen)
        pixel = int(frame_info.enPixelType)
        if w <= 0 or h <= 0 or nlen <= 0:
            raise RuntimeError("Invalid frame meta w=%s h=%s len=%s" % (w, h, nlen))

        self._ensure_bgr_buf(w, h)
        raw = np.frombuffer(self._buf, dtype=np.uint8, count=nlen)
        bgr = self._decode_to_bgr(sdk, raw, w, h, nlen, pixel, frame_info)
        self._update_diag(w, h, nlen, pixel, bgr, path="raw")
        return bgr

    def _update_diag(self, w, h, nlen, pixel, bgr, path: str) -> None:
        mean = float(np.mean(bgr))
        family = classify_pixel_type(pixel) if pixel else self._sensor_family
        self.last_diag.update(
            {
                "width": w,
                "height": h,
                "frame_len": nlen,
                "pixel_type": "0x%x" % int(pixel),
                "mean": round(mean, 2),
                "grab_path": path,
                "sensor_family": family or self._sensor_family,
                "is_mono": (family or self._sensor_family) == "mono",
            }
        )
        if self._fixed_exposure_locked:
            self._dark_streak = 0
        elif mean < 3.0:
            self._dark_streak += 1
            if self._dark_streak == 15:
                self._force_manual_exposure(exposure_us=30000.0, gain_db=12.0)
            elif self._dark_streak == 40:
                self._force_manual_exposure(exposure_us=50000.0, gain_db=15.0)
        else:
            self._dark_streak = 0

    def _decode_to_bgr(self, sdk, raw, w, h, nlen, pixel, frame_info) -> np.ndarray:
        if pixel == sdk.PixelType_Gvsp_BGR8_Packed and nlen >= w * h * 3:
            return raw[: w * h * 3].reshape((h, w, 3)).copy()
        if pixel == sdk.PixelType_Gvsp_RGB8_Packed and nlen >= w * h * 3:
            rgb = raw[: w * h * 3].reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if pixel == sdk.PixelType_Gvsp_Mono8 and nlen >= w * h:
            gray = raw[: w * h].reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Mono16: 16-bit little-endian, take high 8 bits
        if pixel == int(sdk.PixelType_Gvsp_Mono16) and nlen >= w * h * 2:
            gray16 = np.frombuffer(self._buf, dtype=np.uint16, count=w * h).reshape((h, w))
            gray = (gray16 >> 8).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Bayer8 via OpenCV
        bayer_cv = {
            int(sdk.PixelType_Gvsp_BayerRG8): cv2.COLOR_BayerRG2BGR,
            int(sdk.PixelType_Gvsp_BayerGR8): cv2.COLOR_BayerGR2BGR,
            int(sdk.PixelType_Gvsp_BayerGB8): cv2.COLOR_BayerGB2BGR,
            int(sdk.PixelType_Gvsp_BayerBG8): cv2.COLOR_BayerBG2BGR,
        }
        if pixel in bayer_cv and nlen >= w * h:
            gray = raw[: w * h].reshape((h, w))
            return cv2.cvtColor(gray, bayer_cv[pixel])

        # YUV422
        if pixel == int(sdk.PixelType_Gvsp_YUV422_YUYV_Packed) and nlen >= w * h * 2:
            yuv = raw[: w * h * 2].reshape((h, w, 2))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
        if pixel == int(sdk.PixelType_Gvsp_YUV422_Packed) and nlen >= w * h * 2:
            yuv = raw[: w * h * 2].reshape((h, w, 2))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)

        # SDK convert for Mono10/12, Bayer10/12 packed, HB, etc.
        dst_size = int(w) * int(h) * 3
        dst_buf = (c_ubyte * dst_size)()
        conv = sdk.MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(conv), 0, sizeof(sdk.MV_CC_PIXEL_CONVERT_PARAM))
        conv.nWidth = w
        conv.nHeight = h
        conv.pSrcData = cast(self._buf, POINTER(c_ubyte))
        conv.nSrcDataLen = nlen
        conv.enSrcPixelType = frame_info.enPixelType
        conv.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed
        conv.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        conv.nDstBufferSize = dst_size

        ret = self._cam.MV_CC_ConvertPixelType(conv)
        if ret != sdk.MV_OK:
            try:
                ret = self._cam.MV_CC_ConvertPixelTypeEx(conv)
            except Exception:  # noqa: BLE001
                pass
        if ret != sdk.MV_OK:
            # Last-chance: mono8-sized buffer as gray (never treat Bayer-sized wrongly if color)
            if self._sensor_family == "mono" and nlen >= w * h:
                gray = np.frombuffer(self._buf, dtype=np.uint8, count=w * h).reshape((h, w))
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            raise RuntimeError(
                "Unsupported pixel type 0x%x convert failed 0x%x (len=%s %sx%s family=%s)"
                % (pixel, ret, nlen, w, h, self._sensor_family)
            )
        out_len = int(getattr(conv, "nDstLen", 0) or dst_size)
        out = np.frombuffer(dst_buf, dtype=np.uint8, count=min(out_len, dst_size))
        if out.size < w * h * 3:
            raise RuntimeError(
                "ConvertPixelType short output %s < %s (pixel=0x%x)"
                % (out.size, w * h * 3, pixel)
            )
        bgr = out[: w * h * 3].reshape((h, w, 3)).copy()
        if float(np.mean(bgr)) < 0.5 and self._sensor_family == "mono" and nlen >= w * h:
            gray = np.frombuffer(self._buf, dtype=np.uint8, count=w * h).reshape((h, w))
            if float(np.mean(gray)) > 1.0:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr

    def stop_and_close(self) -> None:
        if self._cam is None:
            return
        sdk = getattr(self, "_sdk", None)
        try:
            if self._grabbing:
                self._cam.MV_CC_StopGrabbing()
        except Exception:  # noqa: BLE001
            pass
        self._grabbing = False
        try:
            if self._opened:
                self._cam.MV_CC_CloseDevice()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._cam.MV_CC_DestroyHandle()
        except Exception:  # noqa: BLE001
            pass
        self._opened = False
        self._cam = None
        self._buf = None
        self._bgr_buf = None
        self._bgr_capacity = 0
        self._sdk = sdk
