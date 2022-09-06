import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import utils as utils
from agent.encoder import make_encoder



################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


LOG_FREQ = 10000


def make_sac_agent(obs_shape, action_shape, args, teacher_model=None):
    return SacSSAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        use_rot=args.use_rot,
        use_inv=args.use_inv,
        use_curl=args.use_curl,
        use_kd=args.use_kd,
        ss_lr=args.ss_lr,
        ss_update_freq=args.ss_update_freq,
        num_layers=args.num_layers,
        num_shared_layers=args.num_shared_layers,
        num_filters=args.num_filters,
        curl_latent_dim=args.curl_latent_dim,
    )


def make_ppo_agent(obs_shape, action_shape, args, config):
    return PPO(
        state_dim=obs_shape,
        action_dim=action_shape,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        lam=config.lamda,
        K_epochs=config.K_epochs,
        eps_clip=config.eps_clip,
        has_continuous_action_space=config.has_continuous_action_space,
        action_std_init=config.action_std,
        use_gae=args.use_gae
    )


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class ActorCritic(nn.Module):
    """MLP Actor-Critic Network for PPO."""
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            logits = self.actor(state)
            dist = Categorical(logits=logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            logits = self.actor(state)
            dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, num_shared_layers
    ):
        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class RotFunction(nn.Module):
    """MLP for rotation prediction."""
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, h):
        return self.trunk(h)


class InvFunction(nn.Module):
    """MLP for inverse dynamics model."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)


class CURL(nn.Module):
    """Implements CURL, a contrastive learning method"""
    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder 
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim,
        encoder_feature_dim, num_layers, num_filters, num_shared_layers
    ):
        super().__init__()

        self.encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2


class SacSSAgent(object):
    """
    SAC with an auxiliary self-supervised task.
    Based on https://github.com/denisyarats/pytorch_sac_ae
    """
    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        use_rot=False,
        use_inv=False,
        use_curl=False,
        use_kd=False,
        kd_lr=0.1,
        ss_lr=1e-3,
        ss_update_freq=1,
        num_layers=4,
        num_shared_layers=4,
        num_filters=32,
        curl_latent_dim=128,
    ):
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.ss_update_freq = ss_update_freq
        self.use_rot = use_rot
        self.use_inv = use_inv
        self.use_curl = use_curl
        self.curl_latent_dim = curl_latent_dim

        self.use_kd = use_kd
        self.kd_lr = kd_lr

        assert num_layers >= num_shared_layers, 'num shared layers cannot exceed total amount'

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, num_layers
        ).cuda()

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters, num_layers
        ).cuda()

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters, num_layers
        ).cuda()

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # self-supervision
        self.rot = None
        self.inv = None
        self.curl = None
        self.ss_encoder = None

        if use_rot or use_inv:
            self.ss_encoder = make_encoder(
                obs_shape, encoder_feature_dim, num_layers,
                num_filters, num_shared_layers
            ).cuda()
            self.ss_encoder.copy_conv_weights_from(self.critic.encoder, num_shared_layers)
            
            # rotation
            if use_rot:
                self.rot = RotFunction(encoder_feature_dim, hidden_dim).cuda()
                self.rot.apply(weight_init)

            # inverse dynamics
            if use_inv:
                self.inv = InvFunction(encoder_feature_dim, action_shape[0], hidden_dim).cuda()
                self.inv.apply(weight_init)
            
        # curl
        if use_curl:
            self.curl = CURL(obs_shape, encoder_feature_dim,
                self.curl_latent_dim, self.critic, self.critic_target, output_type='continuous').cuda()

        # ss optimizers
        self.init_ss_optimizers(encoder_lr, ss_lr)

        # sac optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def init_ss_optimizers(self, encoder_lr=1e-3, ss_lr=1e-3):
        if self.ss_encoder is not None:
            self.encoder_optimizer =  torch.optim.Adam(
                self.ss_encoder.parameters(), lr=encoder_lr
            )
        if self.use_rot:
            self.rot_optimizer =  torch.optim.Adam(
                self.rot.parameters(), lr=ss_lr
            )
        if self.use_inv:
            self.inv_optimizer =  torch.optim.Adam(
                self.inv.parameters(), lr=ss_lr
            )
        if self.use_curl:
            self.encoder_optimizer =  torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )
            self.curl_optimizer =  torch.optim.Adam(
                self.curl.parameters(), lr=ss_lr
            )
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.ss_encoder is not None:
            self.ss_encoder.train(training)
        if self.rot is not None:
            self.rot.train(training)
        if self.inv is not None:
            self.inv.train(training)
        if self.curl is not None:
            self.curl.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True, teacher_model=None):
        # detach encoder, so we don't update it with the actor loss
        mu, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if self.use_kd:
            teacher_action, _, _, _ = teacher_model.actor(obs, detach_encoder=True)
            kd_loss = F.mse_loss(mu, teacher_action)
            actor_loss = (1 - self.kd_lr) * actor_loss + self.kd_lr * kd_loss

        if self.kd_lr > 0.02:
            self.kd_lr = 0.99 * self.kd_lr

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_rot(self, obs, L=None, step=None):
        assert obs.shape[-1] == 84

        obs, label = utils.rotate(obs)
        h = self.ss_encoder(obs)
        pred_rotation = self.rot(h)
        rot_loss = F.cross_entropy(pred_rotation, label)

        self.encoder_optimizer.zero_grad()
        self.rot_optimizer.zero_grad()
        rot_loss.backward()

        self.encoder_optimizer.step()
        self.rot_optimizer.step()

        if L is not None:
            L.log('train_rot/rot_loss', rot_loss, step)

        return rot_loss.item()

    def update_inv(self, obs, next_obs, action, L=None, step=None):
        assert obs.shape[-1] == 84 and next_obs.shape[-1] == 84

        h = self.ss_encoder(obs)
        h_next = self.ss_encoder(next_obs)

        pred_action = self.inv(h, h_next)
        inv_loss = F.mse_loss(pred_action, action)

        self.encoder_optimizer.zero_grad()
        self.inv_optimizer.zero_grad()
        inv_loss.backward()

        self.encoder_optimizer.step()
        self.inv_optimizer.step()

        if L is not None:
            L.log('train_inv/inv_loss', inv_loss, step)

        return inv_loss.item()

    def update_curl(self, obs_anchor, obs_pos, L=None, step=None, ema=False):
        assert obs_anchor.shape[-1] == 84 and obs_pos.shape[-1] == 84

        z_a = self.curl.encode(obs_anchor)
        z_pos = self.curl.encode(obs_pos, ema=True)
        
        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().cuda()
        curl_loss = F.cross_entropy(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.curl_optimizer.zero_grad()
        curl_loss.backward()

        self.encoder_optimizer.step()
        self.curl_optimizer.step()
        if L is not None:
            L.log('train/curl_loss', curl_loss, step)

        if ema:
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        return curl_loss.item()

    def update(self, replay_buffer, L, step, teacher_model=None):
        if self.use_curl:
            obs, action, reward, next_obs, not_done, curl_kwargs = replay_buffer.sample_curl()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample()
        
        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, teacher_model=teacher_model)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if self.rot is not None and step % self.ss_update_freq == 0:
            self.update_rot(obs, L, step)

        if self.inv is not None and step % self.ss_update_freq == 0:
            self.update_inv(obs, next_obs, action, L, step)

        if self.curl is not None and step % self.ss_update_freq == 0:
            obs_anchor, obs_pos = curl_kwargs["obs_anchor"], curl_kwargs["obs_pos"]
            self.update_curl(obs_anchor, obs_pos, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.rot is not None:
            torch.save(
                self.rot.state_dict(),
                '%s/rot_%s.pt' % (model_dir, step)
            )
        if self.inv is not None:
            torch.save(
                self.inv.state_dict(),
                '%s/inv_%s.pt' % (model_dir, step)
            )
        if self.curl is not None:
            torch.save(
                self.curl.state_dict(),
                '%s/curl_%s.pt' % (model_dir, step)
            )
        if self.ss_encoder is not None:
            torch.save(
                self.ss_encoder.state_dict(),
                '%s/ss_encoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.rot is not None:
            self.rot.load_state_dict(
                torch.load('%s/rot_%s.pt' % (model_dir, step))
            )
        if self.inv is not None:
            self.inv.load_state_dict(
                torch.load('%s/inv_%s.pt' % (model_dir, step))
            )
        if self.curl is not None:
            self.curl.load_state_dict(
                torch.load('%s/curl_%s.pt' % (model_dir, step))
            )
        if self.ss_encoder is not None:
            self.ss_encoder.load_state_dict(
                torch.load('%s/ss_encoder_%s.pt' % (model_dir, step))
            )


class PPO:
    """
    PPO with Generalized Advantage Estimation.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lam, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, use_gae=False):
        
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_gae = use_gae

        self.buffer = utils.RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
                _, value, _ = self.policy_old.evaluate(state, action)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.values.append(value)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
                _, value, _ = self.policy_old.evaluate(state, action)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.values.append(value)

            return action.item()

    def update(self):
        rewards = []
        if self.use_gae:
            # Generalized Advantage Estimation
            gae = 0
            num_steps = len(self.buffer.rewards)

            for i in reversed(range(num_steps)):
                if i == num_steps - 1:
                    nextnonterminal = 1.0 - self.buffer.is_terminals[i]
                    nextvalues = self.buffer.values[i]
                else:
                    nextnonterminal = 1.0 - self.buffer.is_terminals[i + 1]
                    nextvalues = self.buffer.values[i + 1]
                delta = self.buffer.rewards[i] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[i]
                gae = delta + self.gamma * self.lam * nextnonterminal * gae
                rewards.insert(0, gae + self.buffer.values[i])
        else:       
            # Monte Carlo estimate of returns
            discounted_reward = 0

            for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))