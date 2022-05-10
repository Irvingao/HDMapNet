import torch 
import torch.nn as nn
from torchvision.transforms.functional import rotate
import numpy as np
import mmcv
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

class BEVTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 Z=8,  # the range the Z-axis
                 D=4,  # the number of sample points on Z-axis
                 embed_dims=256,
                 rotate_prev_bev=True,
                 only_encoder=False,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_shift=True,
                 **kwargs):
        super(BEVTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.only_encoder = only_encoder

        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.Z = Z
        self.D = D
        self.rotate_prev_bev = rotate_prev_bev
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_shift = use_shift

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.count = 0

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            # nn.LayerNorm(self.embed_dims)
        )

        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def get_reference_points(H, W, Z=8, D=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in decoder.
        Args:
            H, W spatial shape of bev
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, D, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(-1, H, W) / Z

            # zs = torch.arange(1, Z, 2, dtype=dtype,
            #                  device=device).view(-1, 1, 1).expand(-1, H, W)/Z

            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(D, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(D, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H  # ?
            ref_x = ref_x.reshape(-1)[None] / W  # ?
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def forward(self,
                mlvl_feats,
                bev_embed,
                query_embed,
                bev_h,
                bev_w,
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                return_bev=False,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bs = mlvl_feats[0].size(0)

        bev_embed = bev_embed.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.Z, self.D, bs=bs, dim='3d', device=bev_embed.device, dtype=bev_embed.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bs, device=bev_embed.device, dtype=bev_embed.dtype)

        # velocity = kwargs['img_metas'][0]['can_bus'][-3]
        # delta_time = 0.5 # the gap between two continuse samples is 0.5s
        delta_x = kwargs['img_metas'][0]['can_bus'][0]
        delta_y = kwargs['img_metas'][0]['can_bus'][1]

        ego_angle = kwargs['img_metas'][0]['can_bus'][-2] / np.pi * 180
        rotation_angle = kwargs['img_metas'][0]['can_bus'][-1]
        assert bev_h == bev_w and bev_h == 200
        grid_length = 0.512  # one grid in bev represents 0.512 meters
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)

        translation_angle = np.arctan2(delta_y, delta_x)/np.pi*180
        if translation_angle < 0: translation_angle += 360

        bev_angle = ego_angle - translation_angle

        shift_y = translation_length * np.cos(bev_angle/180*np.pi) / grid_length / bev_h
        shift_x = translation_length * np.sin(bev_angle/180*np.pi) / grid_length / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_embed.new_tensor([shift_x, shift_y])

        # if np.abs(rotation_angle)>10:
        #    print(ego_angle, tmp_angle, rotation_angle, bev_angle, translation_length, shift_x, shift_y)

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h*bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)

            if self.rotate_prev_bev:
                num_prev_bev = prev_bev.size(1)
                re_order = [bev_h - i for i in range(1, bev_h + 1)]
                prev_bev = prev_bev.reshape(bev_h, bev_w, -1)[re_order].permute(2, 0, 1)
                # By considering the visualization results and basic physical knowledge,
                #  I am sure this negative sign is Absolutely correct
                prev_bev = rotate(prev_bev, -rotation_angle)
                prev_bev = prev_bev.permute(1, 2, 0)[re_order].reshape(bev_h * bev_w, num_prev_bev, -1)

        can_bus = bev_embed.new_tensor(kwargs['img_metas'][0]['can_bus'])[None, None, :]
        can_bus = self.can_bus_mlp(can_bus)
        bev_embed = bev_embed + can_bus*self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_embed,
            feat_flatten,
            feat_flatten,
            ref_2d=ref_2d,
            ref_3d=ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        if self.only_encoder:
            return bev_embed

        '''
        if kwargs['img_metas'][0]['sample_idx'] == 'b6c420c3a5bd4a219b1cb82ee5ea0aa7':
            re_order = [bev_h-i for i in range(1, bev_h+1)]
            save_tensor(bev_embed.reshape(bev_h, bev_w, -1)[re_order].permute(2, 0, 1), '{i}.png'.format(i=kwargs['img_metas'][0]['sample_idx']))
            if prev_bev is not None:
                save_tensor(prev_bev.reshape(bev_h, bev_w, -1)[re_order].permute(2, 0, 1), '{i}.png'.format(i=kwargs['img_metas'][0]['prev_idx']))
            self.count+=1
        '''

        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        if return_bev:
            return bev_embed, (inter_states, init_reference_out, inter_references_out)
        else:
            return inter_states, init_reference_out, inter_references_out
        
        
class BEVTransformerEncoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_pre_bev_layer=3, return_intermediate=False, **kwargs):

        super(BEVTransformerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.num_pre_bev_layer = num_pre_bev_layer

    def forward(self,
                query,
                key,
                value,
                *args,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                reg_branches=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = query
        intermediate = []

        reference_points_cam, mask = point_sampling(ref_3d, self.pc_range, query.device, kwargs['img_metas'])

        shift_ref_2d = ref_2d
        shift_ref_2d[..., :] += shift

        for lid, layer in enumerate(self.layers):

            if lid >= self.num_pre_bev_layer:
                prev_bev = None
                cur_ref_2d = ref_2d
            else:
                cur_ref_2d = shift_ref_2d
            output = layer(
                query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=cur_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                mask=mask,
                bev_queue=prev_bev,
                **kwargs)

            query = output

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class BEVEncoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_queue=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                if bev_queue is not None:
                    temp_key = temp_value = bev_queue
                else:
                    temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)

                if bev_queue is not None and query.size(1) > 1:
                    assert  False
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1


            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class BEVCrossDeformableAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MultiScaleDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4)
                 ):
        super(BEVCrossDeformableAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        # self.num_heads = num_heads
        # self.num_points = num_points
        self.num_cams = num_cams

        self.attention_weights = None
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.batch_first = batch_first

 
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        num_query, bs, _ = query.size()

        # reference_points_cam, mask= =point_sampling(reference_points, self.pc_range, kwargs['img_metas'])

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros([max_len, bs * self.num_cams, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs * self.num_cams, max_len, D, 2])
        for i, reference_points_per_img in enumerate(reference_points_cam):
            assert bs == 1, 'batch size must be 1'
            index_query_per_img = indexes[i]
            queries_rebatch[:len(index_query_per_img), i:i + 1] = query[index_query_per_img]
            reference_points_rebatch[i:i + 1, :len(index_query_per_img)] = reference_points_per_img[:,
                                                                           index_query_per_img]

        # _, l, _, _ = key.shape
        l = key.size(1)
        key = key.permute(1, 0, 2, 3).view(l, self.num_cams * bs, self.embed_dims)
        value = value.permute(1, 0, 2, 3).view(l, self.num_cams * bs, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch, key=key, value=value,
                                            reference_points=reference_points_rebatch, spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index)

        for i, index_query_per_img in enumerate(indexes):
            slots[index_query_per_img] += queries[:len(index_query_per_img), i:i + 1]
        count = mask.sum(-1) > 0
        count = count.permute(2, 1, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


# @run_time('point_sampling')
def point_sampling(reference_points, pc_range, device, img_metas):
    # file_names = img_metas[0]['filename']

    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    #
    res = np.matmul(lidar2img.cpu().numpy(), reference_points.cpu().numpy())
    reference_points_cam = torch.from_numpy(res).squeeze(-1).to(device)

    # reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    # reference_points_cam = torch.matmul(lidar2img.cpu(), reference_points.cpu()).to(device).squeeze(-1)
    #

    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    # print('reference_points_cam[..., 0:2]',reference_points_cam[..., 0:2])
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    # reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > 0.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0))
    mask = torch.nan_to_num(mask)

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    mask = mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, mask

class MultiScaleDeformableAttention3D(BaseModule):
    """An attention module used in Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)



    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            # N_levels 2
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            # BS, N_query, N_Head, N_level, N_points, (x,y)

            # BS, N_query, _, N_level, _(num_height), (x,y)
            reference_points = reference_points[:, :, None, :, None, :]
            s1, s2, s3, s4, s5, s6 = reference_points.shape

            reference_points = reference_points.view(s1, s2, s3, s5, 1, s4, s6)
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            c1, c2, c3, c4, c5, c6 = sampling_offsets.shape
            assert c4 == self.num_levels
            sampling_offsets = sampling_offsets.view(c1, c2, c3, c4, c5 // s4, s4, c6)
            sampling_locations = reference_points + sampling_offsets
            t1, t2, t3, t4, t5, t6, t7 = sampling_locations.shape
            sampling_locations = sampling_locations.view(t1, t2, t3, t4, t5 * t6, t7)


        elif reference_points.shape[-1] == 4:
            assert False
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points \
                                 * reference_points[:, :, None, :, None, 2:] \
                                 * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return output