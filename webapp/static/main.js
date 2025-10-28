// Cybernetic Perception Lenses JavaScript
// Handles lens interactions and visual effects

document.addEventListener('alpine:init', () => {
  // Initialize Alpine.js components if needed
  console.log('Perception Lab initialized');
});

// Lens event handlers - toggle body classes for global visual effects
document.addEventListener('lens-deglare', e => {
  document.body.classList.toggle('deglare', e.detail);
  console.log('Deglare lens:', e.detail ? 'activated' : 'deactivated');
});

document.addEventListener('lens-uncertainty', e => {
  document.body.classList.toggle('uncertainty', e.detail);
  console.log('Uncertainty lens:', e.detail ? 'activated' : 'deactivated');
});

document.addEventListener('lens-outliers', e => {
  document.body.classList.toggle('outliers', e.detail);
  console.log('Outliers lens:', e.detail ? 'activated' : 'deactivated');
});

// Enhanced interaction feedback
document.addEventListener('DOMContentLoaded', () => {
  // Add subtle hover effects to perception controls
  const lenses = document.querySelectorAll('.lens');
  lenses.forEach(lens => {
    lens.addEventListener('mouseenter', () => {
      lens.style.transform = 'scale(1.05)';
    });
    lens.addEventListener('mouseleave', () => {
      lens.style.transform = 'scale(1)';
    });
  });

  // Add visual feedback for confidence/anomaly sliders
  const sliders = document.querySelectorAll('input[type="range"]');
  sliders.forEach(slider => {
    slider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      const percentage = (value * 100);
      
      // Visual feedback based on value
      if (e.target.parentElement.textContent.includes('Confidence')) {
        e.target.style.background = `linear-gradient(90deg, #007AFF ${percentage}%, rgba(0,0,0,0.1) ${percentage}%)`;
      } else if (e.target.parentElement.textContent.includes('Anomaly')) {
        e.target.style.background = `linear-gradient(90deg, #FF8C00 ${percentage}%, rgba(0,0,0,0.1) ${percentage}%)`;
      }
    });
  });

  // Initialize slider backgrounds
  sliders.forEach(slider => {
    slider.dispatchEvent(new Event('input'));
  });
});

// Perception state management
window.PerceptionLab = {
  state: {
    uncertainty: false,
    deglare: false,
    outliers: false,
    confidence: 0.0,
    anomaly: 0.0
  },
  
  toggleLens: function(lens) {
    this.state[lens] = !this.state[lens];
    document.body.classList.toggle(lens, this.state[lens]);
    return this.state[lens];
  },
  
  updateFilter: function(filter, value) {
    this.state[filter] = value;
    this.applyFilters();
  },
  
  applyFilters: function() {
    const cards = document.querySelectorAll('[data-conf]');
    cards.forEach(card => {
      const conf = parseFloat(card.dataset.conf) || 0;
      const anom = parseFloat(card.dataset.anom) || 0;
      
      const show = conf >= this.state.confidence && anom >= this.state.anomaly;
      card.style.display = show ? '' : 'none';
      
      if (show && anom < this.state.anomaly) {
        card.style.opacity = '0.35';
      } else {
        card.style.opacity = '1';
      }
    });
  }
};

// Export for global use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = window.PerceptionLab;
}