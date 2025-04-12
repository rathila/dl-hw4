from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):

  class Block(torch.nn.Module):
    def __init__(self, dim, pool=True) -> None:
      super().__init__()
      self.model = torch.nn.Sequential(
          torch.nn.Linear(dim, dim),
          torch.nn.LayerNorm(dim),
          torch.nn.ReLU(),
          torch.nn.Linear(dim, dim),
          torch.nn.LayerNorm(dim)          
      )
      self.relu = torch.nn.ReLU()
     
    def forward(self, x) -> None: 
      identity = x     
      return self.relu(identity + self.model(x))

  def __init__(
      self,
      n_track: int = 10,
      n_waypoints: int = 3,
  ):
      """
      Args:
          n_track (int): number of points in each side of the track
          n_waypoints (int): number of waypoints to predict
      """
      super().__init__()
      hidden_dim = 128
      num_of_blocks = 8
      self.n_track = n_track
      self.n_waypoints = n_waypoints
      input_dim = n_track * 2 * 2  # 10 points * 2 lanes * 2 coords (x, y)
      
      output_dim = n_waypoints * 2  # 3 waypoints * 2 coords
      self.network =  nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Sequential(
          *[self.Block(hidden_dim) for _ in range(num_of_blocks)]
      )
      ,nn.Linear(hidden_dim, output_dim))
      


  def forward(
      self,
      track_left: torch.Tensor,
      track_right: torch.Tensor,
      **kwargs,
  ) -> torch.Tensor:
      """
      Predicts waypoints from the left and right boundaries of the track.

      During test time, your model will be called with
      model(track_left=..., track_right=...), so keep the function signature as is.

      Args:
          track_left (torch.Tensor): shape (b, n_track, 2)
          track_right (torch.Tensor): shape (b, n_track, 2)

      Returns:
          torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
      """
      B = track_left.size(0)
      left_flat = track_left.flatten(start_dim=1)
      right_flat = track_right.flatten(start_dim=1)
      x = torch.cat([left_flat, right_flat], dim=1)    
     
      out = self.network (x)
      return out.view(B, -1, 2)  # (B,



class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # === Input encoder ===
        # Each (x, y) → d_model
        # Input encoder for lane boundaries
        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
           
        )

        # === Transformer Decoder (stacked layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True,dim_feedforward=256)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        # Positional encoding (optional): helps with spatial order
        self.pos_embed = nn.Parameter(torch.randn(20, d_model))  # 10 left + 10 right
        # self.network = torch.nn.Sequential(
        # self.query_embed,
        #   *[TransformerLayer(d_model, 4) for _ in range(4)
        # ],
        #  torch.nn.Linear(d_model, 2))
        # Output projection to (x, y) coordinates
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.size(0)
        # === Step 1: Concatenate and embed track boundaries ===
        track_input = torch.cat([track_left, track_right], dim=1)  # [B, 20, 2]
        memory = self.input_embed(track_input)

        # === Step 2: Expand learnable query embeddings ===
        query_indices = torch.arange(self.n_waypoints, device=track_left.device).unsqueeze(0).repeat(B, 1)  # [B, 3]
        tgt = self.query_embed(query_indices)     # [B, 3, d_model]


        # === Step 3: Transformer Decoder ===
        output = self.transformer_decoder(tgt=tgt, memory=memory)  # [B, 3, d_model]
        

        # === Step 4: Project to 2D waypoint coordinates ===
        waypoints = self.output_projection(output)           
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
