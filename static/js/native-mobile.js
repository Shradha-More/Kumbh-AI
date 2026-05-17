/* ═══════════════════════════════════════════════════════════════
   NATIVE MOBILE — PWA-grade native app behaviors
   Pull-to-refresh, drag-to-dismiss, keyboard avoidance,
   theme-color per page, back gesture, haptics, standalone mode
   ═══════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  var isMobile = function () { return window.innerWidth <= 768; };
  var isStandalone = window.matchMedia('(display-mode: standalone)').matches ||
                     window.navigator.standalone === true;

  // ═══════════════════════════════════════════════════════════
  // KEYBOARD AVOIDANCE
  // Detect virtual keyboard open/close and adjust layout
  // ═══════════════════════════════════════════════════════════

  function initKeyboardAvoidance() {
    if (!isMobile()) return;

    // Use visualViewport API (best approach)
    if (window.visualViewport) {
      var initialHeight = window.visualViewport.height;
      var keyboardOpen = false;

      window.visualViewport.addEventListener('resize', function () {
        var currentHeight = window.visualViewport.height;
        var heightDiff = initialHeight - currentHeight;

        if (heightDiff > 100 && !keyboardOpen) {
          keyboardOpen = true;
          document.body.classList.add('keyboard-open');
          document.body.style.setProperty('--keyboard-height', heightDiff + 'px');

          // Scroll active input into view
          var focused = document.activeElement;
          if (focused && (focused.tagName === 'INPUT' || focused.tagName === 'TEXTAREA')) {
            setTimeout(function () {
              focused.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
          }
        } else if (heightDiff < 100 && keyboardOpen) {
          keyboardOpen = false;
          document.body.classList.remove('keyboard-open');
          document.body.style.removeProperty('--keyboard-height');
        }
      });
    } else {
      // Fallback: detect focus/blur on inputs
      document.addEventListener('focusin', function (e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
          setTimeout(function () {
            document.body.classList.add('keyboard-open');
          }, 300);
        }
      });

      document.addEventListener('focusout', function () {
        setTimeout(function () {
          if (!document.activeElement ||
              (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA')) {
            document.body.classList.remove('keyboard-open');
          }
        }, 100);
      });
    }
  }


  // ═══════════════════════════════════════════════════════════
  // PULL-TO-REFRESH
  // Native-style pull-to-refresh on scrollable pages
  // ═══════════════════════════════════════════════════════════

  function initPullToRefresh() {
    if (!isMobile()) return;

    var pullPages = ['page-home', 'page-explore', 'page-emergency', 'page-profile'];
    var threshold = 80;
    var maxPull = 120;

    pullPages.forEach(function (pageId) {
      var page = document.getElementById(pageId);
      if (!page) return;

      // Create pull indicator
      var indicator = document.createElement('div');
      indicator.className = 'pull-indicator';
      indicator.innerHTML = '<div class="pull-spinner"></div>';
      page.insertBefore(indicator, page.firstChild);

      var startY = 0;
      var currentY = 0;
      var pulling = false;
      var refreshing = false;

      page.addEventListener('touchstart', function (e) {
        if (refreshing) return;
        if (page.scrollTop > 5) return; // Only at top
        startY = e.touches[0].clientY;
        pulling = false;
      }, { passive: true });

      page.addEventListener('touchmove', function (e) {
        if (refreshing) return;
        if (page.scrollTop > 5) { pulling = false; return; }

        currentY = e.touches[0].clientY;
        var delta = currentY - startY;

        if (delta > 10) {
          pulling = true;
          // Rubber-band resistance
          var progress = Math.min(delta / maxPull, 1);
          var rubberBand = progress * (1 - progress * 0.5);
          var height = Math.min(delta * rubberBand, maxPull);

          indicator.style.height = height + 'px';
          indicator.classList.add('active');

          if (delta >= threshold) {
            indicator.classList.add('pulling');
          } else {
            indicator.classList.remove('pulling');
          }

          // Rotate spinner based on pull distance
          var spinner = indicator.querySelector('.pull-spinner');
          if (spinner) {
            var rotation = progress * 360;
            spinner.style.transform = 'scale(' + Math.min(progress * 1.5, 1) + ') rotate(' + rotation + 'deg)';
            spinner.style.opacity = Math.min(progress * 2, 1);
          }
        }
      }, { passive: true });

      page.addEventListener('touchend', function () {
        if (!pulling) return;

        var delta = currentY - startY;
        if (delta >= threshold && !refreshing) {
          // Trigger refresh
          refreshing = true;
          indicator.classList.remove('pulling');
          indicator.classList.add('refreshing');
          indicator.style.height = '60px';

          // Haptic feedback
          if (navigator.vibrate) navigator.vibrate([10, 50, 10]);

          // Perform refresh based on page
          var refreshAction;
          if (pageId === 'page-home') {
            refreshAction = function () {
              if (typeof checkHealth === 'function') checkHealth();
            };
          } else if (pageId === 'page-explore') {
            refreshAction = function () {
              if (typeof loadPlaces === 'function') loadPlaces();
            };
          } else if (pageId === 'page-profile') {
            refreshAction = function () {
              if (typeof updateAuthUI === 'function') updateAuthUI();
              if (typeof loadProfileStats === 'function') loadProfileStats();
              if (typeof loadProfileServer === 'function') loadProfileServer();
            };
          }

          if (refreshAction) refreshAction();

          // End refresh after delay
          setTimeout(function () {
            indicator.classList.remove('active', 'refreshing');
            indicator.style.height = '0';
            var spinner = indicator.querySelector('.pull-spinner');
            if (spinner) {
              spinner.style.transform = '';
              spinner.style.opacity = '';
            }
            refreshing = false;
          }, 1000);
        } else {
          // Cancel pull
          indicator.classList.remove('active', 'pulling');
          indicator.style.height = '0';
          var spinner = indicator.querySelector('.pull-spinner');
          if (spinner) {
            spinner.style.transform = '';
            spinner.style.opacity = '';
          }
        }

        pulling = false;
      }, { passive: true });
    });
  }


  // ═══════════════════════════════════════════════════════════
  // BOTTOM SHEET DRAG-TO-DISMISS
  // Swipe down on bottom sheets to close them
  // ═══════════════════════════════════════════════════════════

  function initDragToDismiss() {
    if (!isMobile()) return;

    var sheets = [
      { overlay: 'locPickerOverlay', close: closeLocPicker },
      { overlay: 'feedbackOverlay', close: closeFeedbackModal },
    ];

    sheets.forEach(function (config) {
      var overlay = document.getElementById(config.overlay);
      if (!overlay) return;

      var sheet = overlay.querySelector('.loc-picker-sheet');
      if (!sheet) return;

      var startY = 0;
      var currentTranslate = 0;
      var dragging = false;

      sheet.addEventListener('touchstart', function (e) {
        // Only start drag from the handle area (top 40px)
        var rect = sheet.getBoundingClientRect();
        var touchY = e.touches[0].clientY - rect.top;
        if (touchY > 40) return;

        startY = e.touches[0].clientY;
        dragging = true;
        sheet.style.transition = 'none';
      }, { passive: true });

      sheet.addEventListener('touchmove', function (e) {
        if (!dragging) return;
        var delta = e.touches[0].clientY - startY;
        if (delta < 0) delta = 0; // Only allow downward drag

        currentTranslate = delta;
        sheet.style.transform = 'translateY(' + delta + 'px)';

        // Dim overlay based on drag distance
        var progress = Math.min(delta / 200, 1);
        overlay.style.background = 'rgba(0,0,0,' + (0.5 * (1 - progress)) + ')';
      }, { passive: true });

      sheet.addEventListener('touchend', function () {
        if (!dragging) return;
        dragging = false;

        sheet.style.transition = 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)';
        overlay.style.transition = 'background 0.3s ease';

        if (currentTranslate > 100) {
          // Dismiss
          sheet.style.transform = 'translateY(100%)';
          if (navigator.vibrate) navigator.vibrate(5);
          setTimeout(function () {
            if (typeof config.close === 'function') config.close();
            sheet.style.transform = '';
            sheet.style.transition = '';
            overlay.style.background = '';
            overlay.style.transition = '';
          }, 300);
        } else {
          // Snap back
          sheet.style.transform = 'translateY(0)';
          overlay.style.background = '';
          setTimeout(function () {
            sheet.style.transition = '';
            overlay.style.transition = '';
          }, 300);
        }

        currentTranslate = 0;
      }, { passive: true });
    });
  }


  // ═══════════════════════════════════════════════════════════
  // THEME-COLOR PER PAGE
  // Changes the status bar color based on active page
  // ═══════════════════════════════════════════════════════════

  function initThemeColorPerPage() {
    var themeColorMap = {
      'home': '#4A0A12',       // deep maroon (Kumbh hero)
      'assistant': '#FFFFFF',   // white (chat)
      'explore': '#0D47A1',     // Godavari blue
      'emergency': '#B71C1C',   // deep red
      'map': '#FFFFFF',         // white
      'login': '#4A0A12',      // deep maroon
      'register': '#4A0A12',   // deep maroon
      'profile': '#1A0A06',    // sacred dark
    };

    var metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) return;

    // Patch navigate to update theme color
    if (typeof window.navigate !== 'function') return;

    var _prevNavigate = window.navigate;
    window.navigate = function (page) {
      _prevNavigate(page);

      var color = themeColorMap[page] || '#6B0F1A';
      metaThemeColor.setAttribute('content', color);
    };
  }


  // ═══════════════════════════════════════════════════════════
  // BACK GESTURE (Android hardware back button)
  // ═══════════════════════════════════════════════════════════

  function initBackButton() {
    var pageHistory = ['home'];

    // Track navigation
    if (typeof window.navigate !== 'function') return;

    var _prevNav = window.navigate;
    window.navigate = function (page) {
      if (pageHistory[pageHistory.length - 1] !== page) {
        pageHistory.push(page);
        // Push browser history state for back button
        try {
          history.pushState({ page: page }, '', '');
        } catch (e) { /* ignore */ }
      }
      _prevNav(page);
    };

    window.addEventListener('popstate', function (e) {
      // Close any open overlays first
      var overlay = document.querySelector('.chat-sidebar-overlay.open');
      if (overlay) {
        if (typeof closeChatSidebar === 'function') closeChatSidebar();
        history.pushState(null, '', '');
        return;
      }

      var locPicker = document.getElementById('locPickerOverlay');
      if (locPicker && locPicker.classList.contains('open')) {
        if (typeof closeLocPicker === 'function') closeLocPicker();
        history.pushState(null, '', '');
        return;
      }

      var feedbackOverlay = document.getElementById('feedbackOverlay');
      if (feedbackOverlay && feedbackOverlay.classList.contains('open')) {
        if (typeof closeFeedbackModal === 'function') closeFeedbackModal();
        history.pushState(null, '', '');
        return;
      }

      // Navigate back
      pageHistory.pop();
      var prevPage = pageHistory[pageHistory.length - 1] || 'home';
      _prevNav(prevPage);
    });

    // Initial state
    try {
      history.replaceState({ page: 'home' }, '', '');
    } catch (e) { /* ignore */ }
  }


  // ═══════════════════════════════════════════════════════════
  // SMOOTH TAB BAR SWITCHING
  // Adds smooth icon transitions when switching tabs
  // ═══════════════════════════════════════════════════════════

  function initTabBarEnhancements() {
    if (!isMobile()) return;

    var tabButtons = document.querySelectorAll('.mobile-nav button');

    tabButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        // Animate the icon
        var icon = this.querySelector('i');
        if (icon) {
          icon.style.transform = 'scale(0.7)';
          icon.style.transition = 'transform 0.15s ease';
          setTimeout(function () {
            icon.style.transform = 'scale(1.15)';
            setTimeout(function () {
              icon.style.transform = 'scale(1)';
              icon.style.transition = 'transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)';
            }, 100);
          }, 80);
        }

        // Haptic feedback
        if (navigator.vibrate) navigator.vibrate(3);
      });
    });
  }


  // ═══════════════════════════════════════════════════════════
  // STANDALONE MODE ENHANCEMENTS
  // Extra behaviors when installed as PWA
  // ═══════════════════════════════════════════════════════════

  function initStandaloneMode() {
    if (!isStandalone) return;

    document.body.classList.add('pwa-standalone');

    // Persist current page across app restarts
    if (typeof window.navigate !== 'function') return;

    var _prevNav = window.navigate;
    window.navigate = function (page) {
      _prevNav(page);
      try {
        sessionStorage.setItem('yatri_page', page);
      } catch (e) { /* ignore */ }
    };

    // Restore page on launch
    try {
      var savedPage = sessionStorage.getItem('yatri_page');
      if (savedPage && savedPage !== 'home') {
        setTimeout(function () {
          _prevNav(savedPage);
        }, 100);
      }
    } catch (e) { /* ignore */ }
  }


  // ═══════════════════════════════════════════════════════════
  // EDGE-SWIPE BACK GESTURE (iOS-like)
  // Swipe from left edge to go back
  // ═══════════════════════════════════════════════════════════

  function initEdgeSwipeBack() {
    if (!isMobile()) return;

    var edgeWidth = 20;
    var threshold = 80;
    var startX = 0;
    var startY = 0;
    var swiping = false;
    var backIndicator = null;

    // Create back indicator
    backIndicator = document.createElement('div');
    backIndicator.style.cssText =
      'position:fixed;top:50%;left:-40px;transform:translateY(-50%);' +
      'width:32px;height:32px;border-radius:50%;z-index:9999;' +
      'background:rgba(232,101,43,0.9);display:flex;align-items:center;' +
      'justify-content:center;box-shadow:0 2px 12px rgba(0,0,0,0.2);' +
      'transition:left 0.1s linear,opacity 0.2s ease;opacity:0;pointer-events:none;';
    backIndicator.innerHTML = '<i class="ri-arrow-left-s-line" style="color:white;font-size:20px"></i>';
    document.body.appendChild(backIndicator);

    document.addEventListener('touchstart', function (e) {
      var touchX = e.touches[0].clientX;
      if (touchX > edgeWidth) return;
      // Don't trigger on chat sidebar or map
      if (e.target.closest('.chat-sidebar, #mapContainer, .route-panel')) return;

      startX = touchX;
      startY = e.touches[0].clientY;
      swiping = true;
    }, { passive: true });

    document.addEventListener('touchmove', function (e) {
      if (!swiping) return;

      var deltaX = e.touches[0].clientX - startX;
      var deltaY = e.touches[0].clientY - startY;

      // Cancel if vertical scroll
      if (Math.abs(deltaY) > Math.abs(deltaX)) {
        swiping = false;
        backIndicator.style.opacity = '0';
        backIndicator.style.left = '-40px';
        return;
      }

      if (deltaX > 10) {
        var progress = Math.min(deltaX / threshold, 1);
        backIndicator.style.opacity = progress;
        backIndicator.style.left = Math.min(deltaX - 16, 20) + 'px';
        backIndicator.style.transform = 'translateY(-50%) scale(' + (0.7 + progress * 0.3) + ')';
      }
    }, { passive: true });

    document.addEventListener('touchend', function (e) {
      if (!swiping) return;
      swiping = false;

      var deltaX = (e.changedTouches[0] ? e.changedTouches[0].clientX : 0) - startX;

      if (deltaX >= threshold) {
        // Trigger back navigation
        if (navigator.vibrate) navigator.vibrate(10);
        history.back();
      }

      backIndicator.style.opacity = '0';
      backIndicator.style.left = '-40px';
      backIndicator.style.transform = 'translateY(-50%) scale(0.7)';
    }, { passive: true });
  }


  // ═══════════════════════════════════════════════════════════
  // SCROLL-AWARE NAVBAR
  // Slightly shrink navbar on scroll for more content space
  // ═══════════════════════════════════════════════════════════

  function initScrollAwareNavbar() {
    if (!isMobile()) return;

    var navbar = document.querySelector('.navbar');
    if (!navbar) return;

    var scrollPages = document.querySelectorAll('#page-home, #page-explore, #page-emergency, #page-profile');

    scrollPages.forEach(function (page) {
      page.addEventListener('scroll', function () {
        var scrollTop = this.scrollTop;
        if (scrollTop > 30) {
          navbar.style.height = 'calc(40px + var(--sat))';
          navbar.style.transition = 'height 0.2s ease';
          navbar.querySelector('.nav-title').style.fontSize = '14px';
          navbar.querySelector('.nav-title').style.transition = 'font-size 0.2s ease';
        } else {
          navbar.style.height = '';
          navbar.querySelector('.nav-title').style.fontSize = '';
        }
      }, { passive: true });
    });
  }


  // ═══════════════════════════════════════════════════════════
  // iOS BOUNCE FIX
  // Prevent body bounce on iOS while allowing inner scroll
  // ═══════════════════════════════════════════════════════════

  function initIOSBounceFix() {
    if (!isMobile()) return;

    document.addEventListener('touchmove', function (e) {
      // Allow scroll inside scrollable containers
      var target = e.target;
      while (target !== document.body && target !== null) {
        var style = window.getComputedStyle(target);
        var overflow = style.overflowY;
        if (overflow === 'auto' || overflow === 'scroll') {
          // Allow scrolling inside this element
          return;
        }
        target = target.parentElement;
      }
      // Prevent body scroll (bounce)
      // Only prevent if not inside a scrollable area
    }, { passive: true });
  }


  // ═══════════════════════════════════════════════════════════
  // WAKE LOCK (keep screen on during voice input)
  // ═══════════════════════════════════════════════════════════

  var wakeLock = null;

  function requestWakeLock() {
    if (!('wakeLock' in navigator)) return;
    navigator.wakeLock.request('screen').then(function (lock) {
      wakeLock = lock;
    }).catch(function () { /* ignore */ });
  }

  function releaseWakeLock() {
    if (wakeLock) {
      wakeLock.release().then(function () { wakeLock = null; });
    }
  }

  // Hook into mic recording
  function initWakeLockOnRecord() {
    var micBtn = document.getElementById('micBtn');
    if (!micBtn) return;

    var observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.attributeName === 'class') {
          if (micBtn.classList.contains('recording')) {
            requestWakeLock();
          } else {
            releaseWakeLock();
          }
        }
      });
    });

    observer.observe(micBtn, { attributes: true });
  }


  // ═══════════════════════════════════════════════════════════
  // ORIENTATION CHANGE HANDLER
  // Smooth transition on device rotation
  // ═══════════════════════════════════════════════════════════

  function initOrientationHandler() {
    window.addEventListener('orientationchange', function () {
      document.body.style.opacity = '0.8';
      document.body.style.transition = 'opacity 0.15s ease';

      setTimeout(function () {
        document.body.style.opacity = '1';

        // Resize particles
        if (window.YatriParticles && typeof window.YatriParticles.resize === 'function') {
          window.YatriParticles.resize();
        }

        // Invalidate map
        if (typeof map !== 'undefined' && map && typeof map.invalidateSize === 'function') {
          map.invalidateSize();
        }
      }, 300);
    });
  }


  // ═══════════════════════════════════════════════════════════
  // INITIALIZE
  // ═══════════════════════════════════════════════════════════

  function init() {
    initKeyboardAvoidance();
    initThemeColorPerPage();
    initBackButton();
    initTabBarEnhancements();
    initStandaloneMode();
    initIOSBounceFix();
    initWakeLockOnRecord();
    initOrientationHandler();

    // Delayed init for gesture-heavy features (ensure DOM is stable)
    setTimeout(function () {
      initPullToRefresh();
      initDragToDismiss();
      initEdgeSwipeBack();
      initScrollAwareNavbar();
    }, 800);

    if (isStandalone) {
      console.log('%c📱 PWA Standalone Mode', 'color: #059669; font-weight: bold;');
    }
    console.log('%c📱 Native Mobile initialized', 'color: #2563EB; font-weight: bold; font-size: 12px;');
  }

  // Public API
  window.NativeMobile = {
    init: init,
    isStandalone: isStandalone,
    requestWakeLock: requestWakeLock,
    releaseWakeLock: releaseWakeLock,
  };

  // Auto-init
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 100);
  }
})();
