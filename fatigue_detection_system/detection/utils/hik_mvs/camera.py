# -*- coding: utf-8 -*-
"""Hikrobot MVS camera helpers: enumerate / open / grab BGR frames."""
from __future__ import annotations

import os
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_MVS_DIR = os.path.dirname(os.path.abspath(__file__))
_MVIMPORT_DIR = os.path.join(_MVS_DIR, "MvImport")
if _MVIMPORT_DIR not in sys.path:
    sys.path.insert(0, _MVIMPORT_DIR)

_sdk_error: Optional[str] = None
_sdk = None


def _load_sdk():
    global _sdk, _sdk_error
    if _sdk is not None:
        return _sdk
    try:
        # Official MVS Samples use flat imports; put MvImport on sys.path.
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
            PixelType_Gvsp_RGB8_Packed,
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
        b.PixelType_Gvsp_BGR8_Packed = PixelType_Gvsp_BGR8_Packed
        b.PixelType_Gvsp_RGB8_Packed = PixelType_Gvsp_RGB8_Packed
        b.PixelType_Gvsp_Mono8 = PixelType_Gvsp_Mono8
        b.PixelType_Gvsp_BayerGR8 = PixelType_Gvsp_BayerGR8
        b.PixelType_Gvsp_BayerRG8 = PixelType_Gvsp_BayerRG8
        b.PixelType_Gvsp_BayerGB8 = PixelType_Gvsp_BayerGB8
        b.PixelType_Gvsp_BayerBG8 = PixelType_Gvsp_BayerBG8
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
    """Open one MVS device and grab BGR numpy frames."""

    def __init__(self):
        self._cam = None
        self._buf = None
        self._payload = 0
        self._opened = False
        self._grabbing = False

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
        # Prefer values from CameraParams if imported on sdk bundle
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
        for name, mode in access_modes:
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

        # Prefer continuous acquisition if trigger mode exists
        try:
            cam.MV_CC_SetEnumValue("TriggerMode", 0)
        except Exception:  # noqa: BLE001
            pass

        # Default imaging: continuous auto exposure + continuous auto gain
        # (runtime only unless user saves UserSet in MVS; applied every app open)
        self._apply_default_auto_exposure_gain(cam, sdk)

        st_param = sdk.MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(sdk.MVCC_INTVALUE))
        ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != sdk.MV_OK or st_param.nCurValue <= 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise RuntimeError("Get PayloadSize failed: 0x%x" % ret)

        self._payload = int(st_param.nCurValue)
        self._buf = (c_ubyte * self._payload)()
        self._cam = cam
        self._sdk = sdk
        self._opened = True

    @staticmethod
    def _set_enum(cam, sdk, key: str, symbolic: str, numeric: int) -> bool:
        """Try GenICam enum by name, then by numeric value. Returns True on success."""
        try:
            ret = cam.MV_CC_SetEnumValueByString(key, symbolic)
            if ret == sdk.MV_OK:
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            ret = cam.MV_CC_SetEnumValue(key, numeric)
            if ret == sdk.MV_OK:
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _apply_default_auto_exposure_gain(self, cam, sdk) -> None:
        """
        Enable continuous auto exposure and continuous auto gain.

        Hikrobot GenICam: ExposureAuto / GainAuto = Continuous (value 2).
        Failures are ignored so unsupported nodes do not block open.
        """
        # MV_EXPOSURE_AUTO_MODE_CONTINUOUS = 2, MV_GAIN_MODE_CONTINUOUS = 2
        continuous = 2
        ok_exp = self._set_enum(cam, sdk, "ExposureAuto", "Continuous", continuous)
        ok_gain = self._set_enum(cam, sdk, "GainAuto", "Continuous", continuous)
        if not ok_gain:
            # Some models expose Gain mode under a different node name
            ok_gain = self._set_enum(cam, sdk, "Gain", "Continuous", continuous)
        # Soft upper bound helps AE recover in dark scenes without crushing highlights
        try:
            cam.MV_CC_SetFloatValue("AutoExposureTimeUpperLimit", 20000.0)
        except Exception:  # noqa: BLE001
            try:
                cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 20000)
            except Exception:  # noqa: BLE001
                pass
        _ = (ok_exp, ok_gain)

    def start_grab(self) -> None:
        if not self._opened:
            raise RuntimeError("Camera not opened")
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != self._sdk.MV_OK:
            raise RuntimeError("StartGrabbing failed: 0x%x" % ret)
        self._grabbing = True

    def get_bgr_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        if not self._grabbing:
            raise RuntimeError("Not grabbing")
        sdk = self._sdk
        frame_info = sdk.MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(sdk.MV_FRAME_OUT_INFO_EX))
        ret = self._cam.MV_CC_GetOneFrameTimeout(self._buf, self._payload, frame_info, timeout_ms)
        if ret != sdk.MV_OK:
            raise RuntimeError("GetOneFrameTimeout failed: 0x%x" % ret)

        w = int(frame_info.nWidth or frame_info.nExtendWidth)
        h = int(frame_info.nHeight or frame_info.nExtendHeight)
        nlen = int(frame_info.nFrameLen)
        pixel = int(frame_info.enPixelType)
        raw = np.frombuffer(self._buf, dtype=np.uint8, count=nlen)

        if pixel == sdk.PixelType_Gvsp_BGR8_Packed:
            return raw.reshape((h, w, 3)).copy()
        if pixel == sdk.PixelType_Gvsp_RGB8_Packed:
            rgb = raw.reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if pixel == sdk.PixelType_Gvsp_Mono8:
            gray = raw.reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Convert Bayer / others to BGR8 via SDK
        dst_size = w * h * 3
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
            raise RuntimeError(
                "Unsupported pixel type 0x%x and convert failed 0x%x" % (pixel, ret)
            )
        out = np.frombuffer(dst_buf, dtype=np.uint8, count=conv.nDstLen or dst_size)
        return out.reshape((h, w, 3)).copy()

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
        self._sdk = sdk
