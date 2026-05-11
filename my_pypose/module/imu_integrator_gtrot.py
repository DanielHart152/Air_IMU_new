import torch
from torch import nn
from ..basics import cumprod
from .. import LieTensor, so3, SO3
from .. import identity_SO3, vec2skew


class IMUIntegratorWithGTRot(nn.Module):
    r'''
    IMU integrator that uses ground truth rotation for gravity compensation and velocity calculation.
    
    This integrator differs from IMUPreintegrator by always using the provided ground truth
    rotation for gravity compensation and velocity transformation, rather than the integrated rotation.

    Args:
        pos (``torch.Tensor``, optional): initial position. Default: :obj:`torch.zeros(3)`.
        rot (``pypose.SO3``, optional): initial rotation. Default: :meth:`pypose.identity_SO3`.
        vel (``torch.Tensor``, optional): initial velocity. Default: ``torch.zeros(3)``.
        gravity (``float``, optional): the gravity acceleration. Default: ``9.81007``.
        reset (``bool``, optional): flag to reset the initial states after the :obj:`forward`
            function is called. If ``False``, the integration starts from the last integration.
            Default: ``False``.

    Example:
        >>> import torch
        >>> import pypose as pp
        >>> p = torch.zeros(3)    # Initial Position
        >>> r = pp.identity_SO3() # Initial rotation
        >>> v = torch.zeros(3)    # Initial Velocity
        >>> integrator = pp.module.IMUIntegratorWithGTRot(p, r, v)
        >>> ang = torch.tensor([0.1,0.1,0.1]) # angular velocity
        >>> acc = torch.tensor([0.1,0.1,0.1]) # acceleration
        >>> gt_rot = pp.mat2SO3(torch.eye(3))    # Ground truth rotation
        >>> dt = torch.tensor([0.002])        # Time difference
        >>> states = integrator(dt, ang, acc, gt_rot)
    '''
    def __init__(self, pos = torch.zeros(3),
                       rot = identity_SO3(),
                       vel = torch.zeros(3),
                       gravity = 9.81007,
                       reset = False):
        super().__init__()
        self.reset = reset

        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity
        self.register_buffer('gravity', torch.tensor([0, 0, gravity]), persistent=False)
        self.register_buffer('pos', self._check(pos).clone(), persistent=False)
        self.register_buffer('rot', self._check(rot).clone(), persistent=False)
        self.register_buffer('vel', self._check(vel).clone(), persistent=False)

    def _check(self, obj):
        if obj is not None:
            if len(obj.shape) == 2:
                obj = obj[None, ...]
            elif len(obj.shape) == 1:
                obj = obj[None, None, ...]
        return obj

    def forward(self, dt, gyro, acc, gt_rot:SO3, init_state=None):
        r"""
        Integrate IMU states using ground truth rotation for gravity compensation.

        Args:
            dt (``torch.Tensor``): time interval from last update.
            gyro (``torch.Tensor``): angular rate in IMU body frame.
            acc (``torch.Tensor``): linear acceleration in IMU body frame (raw sensor input with gravity).
            gt_rot (:obj:`pypose.SO3`): ground truth IMU rotation on the body frame (required).
            init_state (``dict``, optional): the initial state of the integration. The dictionary
                should be in form of :obj:`{'pos': torch.Tensor, 'rot': pypose.SO3, 'vel':
                torch.Tensor}`. If not given, the initial state in constructor will be used.

        Shape:
            - input (:obj:`dt`, :obj:`gyro`, :obj:`acc`, :obj:`gt_rot`): This layer supports the 
              input shape with :math:`(B, F, H_{in})`, :math:`(F, H_{in})` and :math:`(H_{in})`, 
              where :math:`B` is the batch size, :math:`F` is the number of frames, and 
              :math:`H_{in}` is the raw sensor signals.

            - output: a :obj:`dict` of integrated state including ``pos``: position,
              ``rot``: rotation, and ``vel``: velocity, each of which has a shape
              :math:`(B, F, H_{out})`.

        Note:
            This integrator always uses the provided ground truth rotation for:
            1. Gravity compensation: a_world = gt_rot @ (acc - g_body)
            2. Velocity calculation: vel is computed in world frame using gt_rot
        """
        assert gt_rot is not None, "Ground truth rotation must be provided"
        assert(0 < len(acc.shape) == len(dt.shape) == len(gyro.shape) <= 3)
        
        acc = self._check(acc)
        gyro = self._check(gyro)
        dt = self._check(dt)
        gt_rot = self._check(gt_rot)
        B = dt.shape[0]

        if init_state is None:
            init_state = {'pos': self.pos, 'rot': self.rot, 'vel': self.vel}

        inte_state = self.integrate(dt, gyro, acc, gt_rot=gt_rot, init_rot=init_state['rot'])
        predict = self.predict(init_state, inte_state)

        if not self.reset:
            self.pos = predict['pos'][..., -1:, :]
            self.rot = predict['rot'][..., -1:, :]
            self.vel = predict['vel'][..., -1:, :]

        return predict

    def integrate(self, dt, gyro, acc, gt_rot:SO3, init_rot:SO3=None):
        r"""
        Integrate the IMU sensor signals using ground truth rotation for gravity compensation.

        Args:
            dt (``torch.Tensor``): time interval from last update.
            gyro (torch.Tensor): angular rate in IMU body frame.
            acc (``torch.Tensor``): linear acceleration in IMU body frame (raw sensor input with gravity).
            gt_rot (:obj:`pypose.SO3`): ground truth IMU rotation on the body frame.
            init_rot (:obj:`pypose.SO3`, optional): the initial orientation of the IMU state.

        Return:
            ``dict``: integrated states including ``Dp``: position increments, ``Dr``: rotation 
            increments, ``Dv``: velocity increments, and ``Dt``: time increments.
        """
        B, F = dt.shape[:2]
        
        # Compute rotation increments from gyroscope
        dr = so3(gyro*dt).Exp()
        w = torch.cat([identity_SO3(B, 1, dtype=dt.dtype, device=dt.device), dr], dim=1)
        incre_r = cumprod(w, dim=1, left=False)

        # Use ground truth rotation for gravity compensation
        # a_world = gt_rot @ (acc_body - g_body)
        a = acc - gt_rot.Inv() @ self.gravity

        # Compute velocity increments using ground truth rotation
        # dv = gt_rot @ a * dt (acceleration in world frame)
        dv = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dv = torch.cat([dv, gt_rot @ a * dt], dim=1)
        incre_v = torch.cumsum(dv, dim=1)

        # Compute position increments using ground truth rotation
        # dp = vel * dt + 0.5 * gt_rot @ a * dt^2
        dp = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dp = torch.cat([dp, incre_v[:,:F,:] * dt + gt_rot @ a * 0.5 * dt**2], dim=1)
        incre_p = torch.cumsum(dp, dim=1)

        incre_t = torch.cumsum(dt, dim=1)
        incre_t = torch.cat([torch.zeros(B, 1, 1, dtype=dt.dtype, device=dt.device), incre_t], dim=1)

        return {
            'Dp': incre_p[:,1:,:], 
            'Dv': incre_v[...,1:,:], 
            'Dr': incre_r[:,1:,:],
            'Dt': incre_t[...,1:,:]
        }

    @classmethod
    def predict(cls, init_state, integrate):
        r"""
        Propagate the next IMU state from the initial IMU state with the integrated measurements.

        Args:
            init_state (``dict``): the initial state of the integration. The dictionary
                should be in form of :obj:`{'pos': torch.Tensor, 'rot': pypose.SO3, 'vel':
                torch.Tensor}`.
            integrate (``dict``): the integrated IMU measurements. The dictionary
                should be in form of :obj:`{'Dp': torch.Tensor, 'Dr': pypose.SO3, 'Dv':
                torch.Tensor, 'Dt': torch.Tensor}`.

        Return:
            ``dict``: integrated states including ``pos``: position, ``rot``: rotation, and
            ``vel``: velocity.
        """
        return {
            'rot': init_state['rot'] * integrate['Dr'],
            'vel': init_state['vel'] + integrate['Dv'],
            'pos': init_state['pos'] + integrate['Dp'] + init_state['vel'] * integrate['Dt'],
        }
