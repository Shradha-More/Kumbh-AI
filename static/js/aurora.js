/* ═══════════════════════════════════════════════════════════════
   AURORA ENGINE — Advanced Animation Controller for Yatri AI
   Scroll reveals, 3D tilt, magnetic buttons, ripples,
   page transitions, counters, parallax, gesture support
   ═══════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ─── CONFIGURATION ───
  const CONFIG = {
    scrollRevealThreshold: 0.12,
    scrollRevealRootMargin: '0px 0px -40px 0px',
    tiltMaxAngle: 8,
    tiltPerspective: 1000,
    magneticStrength: 0.3,
    magneticDistance: 80,
    rippleDuration: 600,
    counterDuration: 1500,
    counterEase: function (t) { return 1 - Math.pow(1 - t, 4); }, // ease-out-quart
    swipeThreshold: 60,
    swipeVelocity: 0.3,
  };

  // ─── STATE ───
  let scrollObserver = null;
  let tiltElements = [];
  let magneticElements = [];
  let rafId = null;
  let isPageTransitioning = false;

  // ═══════════════════════════════════════════════════════════
  // SCROLL-TRIGGERED REVEALS (Intersection Observer)
  // ═══════════════════════════════════════════════════════════

  function initScrollReveals() {
    const elements = document.querySelectorAll('[data-reveal]');
    if (!elements.length) return;

    // Disconnect previous observer if any
    if (scrollObserver) scrollObserver.disconnect();

    scrollObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          scrollObserver.unobserve(entry.target);
        }
      });
    }, {
      threshold: CONFIG.scrollRevealThreshold,
      rootMargin: CONFIG.scrollRevealRootMargin,
    });

    elements.forEach(function (el) {
      scrollObserver.observe(el);
    });
  }

  // ═══════════════════════════════════════════════════════════
  // 3D CARD TILT EFFECT
  // ═══════════════════════════════════════════════════════════

  function initTilt() {
    tiltElements = document.querySelectorAll('[data-tilt]');
    if (window.innerWidth < 768) return; // Skip on mobile

    tiltElements.forEach(function (el) {
      // Add shine overlay if not present
      if (!el.querySelector('.tilt-shine')) {
        var shine = document.createElement('div');
        shine.className = 'tilt-shine';
        el.appendChild(shine);
      }

      el.addEventListener('mousemove', handleTiltMove, { passive: true });
      el.addEventListener('mouseleave', handleTiltLeave, { passive: true });
    });
  }

  function handleTiltMove(e) {
    var el = this;
    var rect = el.getBoundingClientRect();
    var centerX = rect.left + rect.width / 2;
    var centerY = rect.top + rect.height / 2;
    var mouseX = e.clientX - centerX;
    var mouseY = e.clientY - centerY;

    var rotateY = (mouseX / (rect.width / 2)) * CONFIG.tiltMaxAngle;
    var rotateX = -(mouseY / (rect.height / 2)) * CONFIG.tiltMaxAngle;

    el.style.transform =
      'perspective(' + CONFIG.tiltPerspective + 'px) ' +
      'rotateX(' + rotateX + 'deg) ' +
      'rotateY(' + rotateY + 'deg) ' +
      'scale3d(1.02, 1.02, 1.02)';

    // Move shine based on mouse position
    var shine = el.querySelector('.tilt-shine');
    if (shine) {
      var shineX = ((mouseX / (rect.width / 2)) + 1) / 2 * 100;
      var shineY = ((mouseY / (rect.height / 2)) + 1) / 2 * 100;
      shine.style.background =
        'radial-gradient(circle at ' + shineX + '% ' + shineY + '%, ' +
        'rgba(255,255,255,0.25) 0%, rgba(255,255,255,0) 60%)';
    }
  }

  function handleTiltLeave() {
    this.style.transform = 'perspective(' + CONFIG.tiltPerspective + 'px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
    this.style.transition = 'transform 0.5s cubic-bezier(0.16, 1, 0.3, 1)';
    var self = this;
    setTimeout(function () { self.style.transition = ''; }, 500);
  }

  // ═══════════════════════════════════════════════════════════
  // MAGNETIC BUTTON EFFECT
  // ═══════════════════════════════════════════════════════════

  function initMagnetic() {
    magneticElements = document.querySelectorAll('[data-magnetic]');
    if (window.innerWidth < 768) return;

    magneticElements.forEach(function (el) {
      el.addEventListener('mousemove', handleMagneticMove, { passive: true });
      el.addEventListener('mouseleave', handleMagneticLeave, { passive: true });
    });
  }

  function handleMagneticMove(e) {
    var el = this;
    var rect = el.getBoundingClientRect();
    var centerX = rect.left + rect.width / 2;
    var centerY = rect.top + rect.height / 2;
    var deltaX = (e.clientX - centerX) * CONFIG.magneticStrength;
    var deltaY = (e.clientY - centerY) * CONFIG.magneticStrength;

    el.style.transform = 'translate(' + deltaX + 'px, ' + deltaY + 'px)';
  }

  function handleMagneticLeave() {
    this.style.transform = 'translate(0, 0)';
  }

  // ═══════════════════════════════════════════════════════════
  // RIPPLE EFFECT
  // ═══════════════════════════════════════════════════════════

  function initRipples() {
    // Add ripple to all clickable elements
    document.addEventListener('click', function (e) {
      var target = e.target.closest(
        '.action-card, .sos-card, .hero-cta, .auth-btn, .filter-chip, ' +
        '.lang-chip, .chat-welcome-chip, .helpline-item, .mpc-btn, ' +
        '.route-go-btn, .profile-btn, .sos-call-btn, .map-btn, .fb-type, ' +
        '.loc-picker-gps, .loc-picker-done, .chat-sidebar-new, .nav-link, ' +
        '.mobile-nav button'
      );
      if (!target) return;

      createRipple(target, e);
    }, { passive: true });
  }

  function createRipple(el, e) {
    // Ensure element has proper positioning for ripple
    var pos = getComputedStyle(el).position;
    if (pos === 'static') el.style.position = 'relative';
    el.style.overflow = 'hidden';

    var rect = el.getBoundingClientRect();
    var size = Math.max(rect.width, rect.height) * 2;
    var x = e.clientX - rect.left - size / 2;
    var y = e.clientY - rect.top - size / 2;

    var ripple = document.createElement('span');
    ripple.className = 'ripple-wave';

    // Use light ripple on dark backgrounds
    var bgColor = getComputedStyle(el).backgroundColor;
    if (isDarkColor(bgColor)) {
      ripple.classList.add('ripple-light');
    }

    ripple.style.width = size + 'px';
    ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';

    el.appendChild(ripple);

    setTimeout(function () {
      if (ripple.parentNode) ripple.parentNode.removeChild(ripple);
    }, CONFIG.rippleDuration);
  }

  function isDarkColor(color) {
    if (!color || color === 'transparent' || color === 'rgba(0, 0, 0, 0)') return false;
    var match = color.match(/\d+/g);
    if (!match || match.length < 3) return false;
    var brightness = (parseInt(match[0]) * 299 + parseInt(match[1]) * 587 + parseInt(match[2]) * 114) / 1000;
    return brightness < 128;
  }

  // ═══════════════════════════════════════════════════════════
  // ANIMATED COUNTERS
  // ═══════════════════════════════════════════════════════════

  function initCounters() {
    var counters = document.querySelectorAll('[data-counter]');
    if (!counters.length) return;

    var counterObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    counters.forEach(function (el) {
      counterObserver.observe(el);
    });
  }

  function animateCounter(el) {
    var target = parseInt(el.getAttribute('data-counter') || el.textContent, 10);
    if (isNaN(target)) return;

    var start = 0;
    var startTime = null;

    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      var progress = Math.min((timestamp - startTime) / CONFIG.counterDuration, 1);
      var easedProgress = CONFIG.counterEase(progress);
      var current = Math.round(start + (target - start) * easedProgress);

      el.textContent = current.toLocaleString();

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  // ═══════════════════════════════════════════════════════════
  // PAGE TRANSITIONS
  // ═══════════════════════════════════════════════════════════

  function initPageTransitions() {
    // Patch the navigate function to add transition animations
    if (typeof window.navigate !== 'function') return;

    var originalNavigate = window.navigate;

    window.navigate = function (page) {
      if (isPageTransitioning) return;

      var currentPage = document.querySelector('.page.active');
      var targetPage = document.getElementById('page-' + page);

      if (!targetPage || currentPage === targetPage) {
        originalNavigate(page);
        return;
      }

      isPageTransitioning = true;

      // Animate current page out
      if (currentPage) {
        currentPage.classList.add('page-leaving');
      }

      // Short delay then switch
      setTimeout(function () {
        if (currentPage) {
          currentPage.classList.remove('active', 'page-leaving', 'page-entering');
        }

        // Call original navigate to update active states
        originalNavigate(page);

        // Animate new page in
        if (targetPage) {
          targetPage.classList.add('page-entering');

          // Re-observe any new reveal elements
          reinitRevealElements(targetPage);

          // Handle particles
          if (page === 'home') {
            if (window.YatriParticles) window.YatriParticles.start();
          } else {
            if (window.YatriParticles) window.YatriParticles.stop();
          }

          setTimeout(function () {
            targetPage.classList.remove('page-entering');
            isPageTransitioning = false;
          }, 500);
        } else {
          isPageTransitioning = false;
        }
      }, 200);
    };
  }

  function reinitRevealElements(container) {
    if (!scrollObserver || !container) return;

    var elements = container.querySelectorAll('[data-reveal]:not(.revealed)');
    elements.forEach(function (el) {
      el.classList.remove('revealed');
      scrollObserver.observe(el);
    });
  }

  // ═══════════════════════════════════════════════════════════
  // PARALLAX SCROLL
  // ═══════════════════════════════════════════════════════════

  function initParallax() {
    var parallaxElements = document.querySelectorAll('[data-parallax]');
    if (!parallaxElements.length) return;

    var ticking = false;
    window.addEventListener('scroll', function () {
      if (!ticking) {
        requestAnimationFrame(function () {
          updateParallax(parallaxElements);
          ticking = false;
        });
        ticking = true;
      }
    }, { passive: true });
  }

  function updateParallax(elements) {
    var scrollY = window.pageYOffset;

    elements.forEach(function (el) {
      var speed = parseFloat(el.getAttribute('data-parallax')) || 0.3;
      var rect = el.getBoundingClientRect();

      if (rect.bottom > 0 && rect.top < window.innerHeight) {
        var yPos = -(scrollY * speed);
        el.style.transform = 'translateY(' + yPos + 'px)';
      }
    });
  }

  // ═══════════════════════════════════════════════════════════
  // MOBILE SWIPE GESTURES
  // ═══════════════════════════════════════════════════════════

  function initSwipeGestures() {
    if (window.innerWidth > 768) return;

    var startX, startY, startTime;
    var pages = ['home', 'assistant', 'explore', 'map', 'emergency'];

    document.addEventListener('touchstart', function (e) {
      // Don't interfere with chat sidebar, map, or scrollable elements
      if (e.target.closest('.chat-sidebar, #mapContainer, .route-panel, .chat-messages, .places-list, .filter-bar')) return;

      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
      startTime = Date.now();
    }, { passive: true });

    document.addEventListener('touchend', function (e) {
      if (startX === undefined) return;

      var endX = e.changedTouches[0].clientX;
      var endY = e.changedTouches[0].clientY;
      var deltaX = endX - startX;
      var deltaY = endY - startY;
      var elapsed = Date.now() - startTime;
      var velocity = Math.abs(deltaX) / elapsed;

      // Must be primarily horizontal swipe
      if (Math.abs(deltaX) < CONFIG.swipeThreshold || Math.abs(deltaY) > Math.abs(deltaX)) {
        startX = undefined;
        return;
      }

      if (velocity < CONFIG.swipeVelocity) {
        startX = undefined;
        return;
      }

      // Find current page index
      var currentPage = document.querySelector('.page.active');
      if (!currentPage) { startX = undefined; return; }

      var currentId = currentPage.id.replace('page-', '');
      var currentIdx = pages.indexOf(currentId);
      if (currentIdx === -1) { startX = undefined; return; }

      var targetIdx;
      if (deltaX < 0) {
        // Swipe left → next page
        targetIdx = Math.min(currentIdx + 1, pages.length - 1);
      } else {
        // Swipe right → previous page
        targetIdx = Math.max(currentIdx - 1, 0);
      }

      if (targetIdx !== currentIdx && typeof window.navigate === 'function') {
        window.navigate(pages[targetIdx]);
        // Haptic feedback
        if (navigator.vibrate) navigator.vibrate(10);
      }

      startX = undefined;
    }, { passive: true });
  }

  // ═══════════════════════════════════════════════════════════
  // CHAT MESSAGE ANIMATION
  // ═══════════════════════════════════════════════════════════

  function initChatObserver() {
    var chatContainer = document.getElementById('chatMessages');
    if (!chatContainer) return;

    var msgObserver = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        mutation.addedNodes.forEach(function (node) {
          if (node.nodeType === 1 && node.classList.contains('msg')) {
            node.classList.add('msg-animate');
            // Trigger animation on next frame
            requestAnimationFrame(function () {
              requestAnimationFrame(function () {
                node.classList.add('msg-animate');
              });
            });
          }
        });
      });
    });

    msgObserver.observe(chatContainer, { childList: true });
  }

  // ═══════════════════════════════════════════════════════════
  // SMOOTH NUMBER UPDATES (for profile stats, place count)
  // ═══════════════════════════════════════════════════════════

  function observeStatUpdates() {
    var statElements = document.querySelectorAll('#statQueries, #statLangs, #statFavs');
    if (!statElements.length) return;

    var observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === 'characterData' || mutation.type === 'childList') {
          var el = mutation.target.nodeType === 1 ? mutation.target : mutation.target.parentElement;
          if (el) {
            el.style.transform = 'scale(1.3)';
            el.style.transition = 'transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
            setTimeout(function () {
              el.style.transform = 'scale(1)';
            }, 200);
          }
        }
      });
    });

    statElements.forEach(function (el) {
      observer.observe(el, { childList: true, characterData: true, subtree: true });
    });
  }

  // ═══════════════════════════════════════════════════════════
  // SMOOTH INPUT LABEL FLOAT
  // ═══════════════════════════════════════════════════════════

  function initInputAnimations() {
    var inputs = document.querySelectorAll('.auth-field input, .chat-input, .map-search');

    inputs.forEach(function (input) {
      input.addEventListener('focus', function () {
        this.parentElement.style.transform = 'scale(1.01)';
        this.parentElement.style.transition = 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)';
      });
      input.addEventListener('blur', function () {
        this.parentElement.style.transform = 'scale(1)';
      });
    });
  }

  // ═══════════════════════════════════════════════════════════
  // HAPTIC FEEDBACK (Mobile)
  // ═══════════════════════════════════════════════════════════

  function initHaptics() {
    if (!navigator.vibrate) return;

    document.addEventListener('click', function (e) {
      var target = e.target.closest('button, .action-card, .sos-card, .filter-chip, .lang-chip');
      if (target) navigator.vibrate(5);
    }, { passive: true });
  }

  // ═══════════════════════════════════════════════════════════
  // LOADING SKELETON MANAGER
  // ═══════════════════════════════════════════════════════════

  function showSkeletons(container, count, className) {
    if (!container) return;
    var html = '';
    for (var i = 0; i < count; i++) {
      html += '<div class="skeleton ' + (className || 'skeleton-card') + '"></div>';
    }
    container.innerHTML = html;
  }

  function hideSkeletons(container) {
    if (!container) return;
    var skeletons = container.querySelectorAll('.skeleton');
    skeletons.forEach(function (s) {
      s.style.animation = 'auroraRevealUp 0.3s ease reverse forwards';
      setTimeout(function () {
        if (s.parentNode) s.parentNode.removeChild(s);
      }, 300);
    });
  }

  // ═══════════════════════════════════════════════════════════
  // ENHANCED TOAST
  // ═══════════════════════════════════════════════════════════

  function patchToast() {
    if (typeof window.showToast !== 'function') return;

    var originalShowToast = window.showToast;

    window.showToast = function (message, duration) {
      var toast = document.getElementById('toast');
      var toastText = document.getElementById('toastText');
      if (!toast || !toastText) {
        originalShowToast(message, duration);
        return;
      }

      toastText.textContent = message;
      toast.style.display = 'block';
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(-50%) translateY(-20px) scale(0.9)';

      requestAnimationFrame(function () {
        toast.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0) scale(1)';
      });

      setTimeout(function () {
        toast.style.transition = 'all 0.3s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(-20px) scale(0.9)';
        setTimeout(function () { toast.style.display = 'none'; }, 300);
      }, duration || 3000);
    };
  }

  // ═══════════════════════════════════════════════════════════
  // STAGGER ANIMATION FOR DYNAMICALLY LOADED CONTENT
  // ═══════════════════════════════════════════════════════════

  function staggerChildren(container, animationClass, delay) {
    if (!container) return;
    var children = container.children;
    var baseDelay = delay || 60;

    for (var i = 0; i < children.length; i++) {
      var child = children[i];
      child.style.opacity = '0';
      child.style.animationDelay = (i * baseDelay) + 'ms';
      child.classList.add(animationClass || 'revealed');
      child.setAttribute('data-reveal', 'up');
    }

    // Trigger reveals
    requestAnimationFrame(function () {
      for (var i = 0; i < children.length; i++) {
        children[i].classList.add('revealed');
      }
    });
  }

  // ═══════════════════════════════════════════════════════════
  // SCROLL PROGRESS INDICATOR (optional, for long pages)
  // ═══════════════════════════════════════════════════════════

  function initScrollProgress() {
    var indicator = document.createElement('div');
    indicator.style.cssText =
      'position:fixed;top:0;left:0;height:2px;z-index:9999;' +
      'background:linear-gradient(90deg,var(--gold-dark,#A67C00),var(--gold,#D4A017),var(--gold-light,#ECC94B));' +
      'transform-origin:left;transform:scaleX(0);' +
      'transition:transform 0.1s linear;pointer-events:none;width:100%;';
    indicator.id = 'scrollProgress';
    document.body.appendChild(indicator);

    var ticking = false;
    window.addEventListener('scroll', function () {
      if (!ticking) {
        requestAnimationFrame(function () {
          var scrollTop = window.pageYOffset;
          var docHeight = document.documentElement.scrollHeight - window.innerHeight;
          var progress = docHeight > 0 ? scrollTop / docHeight : 0;
          indicator.style.transform = 'scaleX(' + progress + ')';
          ticking = false;
        });
        ticking = true;
      }
    }, { passive: true });
  }

  // ═══════════════════════════════════════════════════════════
  // INITIALIZE EVERYTHING
  // ═══════════════════════════════════════════════════════════

  function init() {
    // Core animations
    initScrollReveals();
    initTilt();
    initMagnetic();
    initRipples();
    initCounters();
    initParallax();

    // Page transitions (must run after navigate is defined)
    setTimeout(function () {
      initPageTransitions();
      patchToast();
    }, 500);

    // Chat enhancements
    initChatObserver();
    observeStatUpdates();

    // Input animations
    initInputAnimations();

    // Mobile features
    initSwipeGestures();
    initHaptics();

    // Scroll progress
    initScrollProgress();

    // Log initialization
    console.log('%c✨ Aurora Engine initialized', 'color: #E8652B; font-weight: bold; font-size: 12px;');
  }

  // Public API
  window.Aurora = {
    init: init,
    initScrollReveals: initScrollReveals,
    staggerChildren: staggerChildren,
    showSkeletons: showSkeletons,
    hideSkeletons: hideSkeletons,
    animateCounter: animateCounter,
    reinitRevealElements: reinitRevealElements,
  };

  // Auto-init
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 50);
  }
})();
