"""
Multi-Task Floorplan Segmentation Architecture for ArchiGen 3D.
Jointly predicts:
1. Structural layout (Background, Wall_Internal, Wall_External, Door, Window) -> 5 classes
2. Semantic Room Types (Background, LivingRoom, Bedroom, Kitchen, Bath, Entry, Outdoor, Storage, Closet, Garage, Office, Other) -> 12 classes
Powered by SMP UNet++ with ImageNet-pretrained encoder.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class MultiTaskFloorplanUNet(nn.Module):
    """
    Multi-task segmentation model with a shared encoder/decoder backbone
    and dual heads for structural elements and room semantic categorization.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        num_struct_classes: int = 5,
        num_room_classes: int = 12,
        in_channels: int = 3
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_struct_classes = num_struct_classes
        self.num_room_classes = num_room_classes

        # Base UNet++ Backbone
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_struct_classes
        )

        decoder_out_channels = self.base.segmentation_head[0].in_channels

        # Head 1: Structural Head (Walls, Doors, Windows, Background)
        self.struct_head = self.base.segmentation_head

        # Head 2: Room Head (Semantic Room Types)
        self.room_head = smp.base.SegmentationHead(
            in_channels=decoder_out_channels,
            out_channels=num_room_classes,
            activation=None,
            kernel_size=3
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.base.encoder(x)
        decoder_output = self.base.decoder(features)

        struct_logits = self.struct_head(decoder_output)
        room_logits = self.room_head(decoder_output)

        return {
            "struct": struct_logits,
            "room": room_logits
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs forward pass and applies argmax for discrete predictions."""
        out = self.forward(x)
        return {
            "struct_pred": torch.argmax(out["struct"], dim=1),
            "room_pred": torch.argmax(out["room"], dim=1)
        }
