/* ═══════════════════════════════════════════════════════════════
   PARTICLE SYSTEM — Interactive Canvas Particles for Hero Section
   GPU-accelerated, responds to mouse/touch, auto-resizes
   ═══════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  const PARTICLE_COUNT_DESKTOP = 60;
  const PARTICLE_COUNT_MOBILE = 30;
  const CONNECTION_DISTANCE = 120;
  const MOUSE_RADIUS = 150;
  const BASE_SPEED = 0.3;

  let canvas, ctx, particles, mouse, animId, isRunning;

  function isMobile() {
    return window.innerWidth < 768;
  }

  function init() {
    canvas = document.getElementById('heroParticles');
    if (!canvas) return;

    ctx = canvas.getContext('2d');
    mouse = { x: -9999, y: -9999 };
    particles = [];
    isRunning = false;

    resize();
    createParticles();
    bindEvents();
    start();
  }

  function resize() {
    if (!canvas) return;
    const hero = canvas.parentElement;
    if (!hero) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = hero.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);
  }

  function createParticles() {
    const count = isMobile() ? PARTICLE_COUNT_MOBILE : PARTICLE_COUNT_DESKTOP;
    const hero = canvas.parentElement;
    if (!hero) return;
    const rect = hero.getBoundingClientRect();
    particles = [];

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * rect.width,
        y: Math.random() * rect.height,
        vx: (Math.random() - 0.5) * BASE_SPEED * 2,
        vy: (Math.random() - 0.5) * BASE_SPEED * 2,
        radius: Math.random() * 2.5 + 1,
        opacity: Math.random() * 0.5 + 0.2,
        // Sacred gold/saffron/warm palette (matching Kumbh logo)
        color: ['rgba(255,255,255,', 'rgba(236,201,75,', 'rgba(212,160,23,', 'rgba(191,75,27,'][Math.floor(Math.random() * 4)],
        pulseSpeed: Math.random() * 0.02 + 0.01,
        pulsePhase: Math.random() * Math.PI * 2,
      });
    }
  }

  function bindEvents() {
    const hero = canvas.parentElement;
    if (!hero) return;

    hero.addEventListener('mousemove', function (e) {
      const rect = hero.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    }, { passive: true });

    hero.addEventListener('mouseleave', function () {
      mouse.x = -9999;
      mouse.y = -9999;
    }, { passive: true });

    hero.addEventListener('touchmove', function (e) {
      const rect = hero.getBoundingClientRect();
      const touch = e.touches[0];
      mouse.x = touch.clientX - rect.left;
      mouse.y = touch.clientY - rect.top;
    }, { passive: true });

    hero.addEventListener('touchend', function () {
      mouse.x = -9999;
      mouse.y = -9999;
    }, { passive: true });

    let resizeTimer;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        resize();
        createParticles();
      }, 250);
    }, { passive: true });

    // Visibility API — pause when hidden
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) stop();
      else start();
    });
  }

  function update() {
    const hero = canvas.parentElement;
    if (!hero) return;
    const rect = hero.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const now = Date.now() * 0.001;

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // Mouse interaction — push particles away gently
      const dx = p.x - mouse.x;
      const dy = p.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < MOUSE_RADIUS && dist > 0) {
        const force = (MOUSE_RADIUS - dist) / MOUSE_RADIUS;
        const angle = Math.atan2(dy, dx);
        p.vx += Math.cos(angle) * force * 0.3;
        p.vy += Math.sin(angle) * force * 0.3;
      }

      // Apply velocity with damping
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.99;
      p.vy *= 0.99;

      // Wrap around edges
      if (p.x < -10) p.x = w + 10;
      if (p.x > w + 10) p.x = -10;
      if (p.y < -10) p.y = h + 10;
      if (p.y > h + 10) p.y = -10;

      // Pulse radius
      p.currentRadius = p.radius + Math.sin(now * p.pulseSpeed * 60 + p.pulsePhase) * 0.5;
      p.currentOpacity = p.opacity + Math.sin(now * p.pulseSpeed * 40 + p.pulsePhase) * 0.1;
    }
  }

  function draw() {
    const hero = canvas.parentElement;
    if (!hero) return;
    const rect = hero.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw connections
    if (!isMobile()) {
      ctx.lineWidth = 0.5;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < CONNECTION_DISTANCE) {
            const alpha = (1 - dist / CONNECTION_DISTANCE) * 0.15;
            ctx.strokeStyle = 'rgba(255,255,255,' + alpha + ')';
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw mouse connections
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < MOUSE_RADIUS) {
          const alpha = (1 - dist / MOUSE_RADIUS) * 0.25;
          ctx.strokeStyle = 'rgba(212,160,23,' + alpha + ')';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();
          ctx.lineWidth = 0.5;
        }
      }
    }

    // Draw particles
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(p.currentRadius, 0.5), 0, Math.PI * 2);
      ctx.fillStyle = p.color + Math.max(p.currentOpacity, 0) + ')';
      ctx.fill();
    }
  }

  function loop() {
    if (!isRunning) return;
    update();
    draw();
    animId = requestAnimationFrame(loop);
  }

  function start() {
    if (isRunning) return;
    // Check if hero is visible (page-home is active)
    const pageHome = document.getElementById('page-home');
    if (pageHome && !pageHome.classList.contains('active')) return;

    isRunning = true;
    loop();
  }

  function stop() {
    isRunning = false;
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  }

  // Public API
  window.YatriParticles = {
    init: init,
    start: start,
    stop: stop,
    resize: function () { resize(); createParticles(); },
  };

  // Auto-init when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // Small delay to ensure hero section is rendered
    setTimeout(init, 100);
  }
})();
