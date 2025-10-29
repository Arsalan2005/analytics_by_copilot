function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

document.addEventListener('DOMContentLoaded', () => {
  const runButton = document.querySelector('[data-run-analyses]');
  const spinner = runButton?.querySelector('[data-spinner]');
  const label = runButton?.querySelector('[data-label]');
  const feedback = document.querySelector('[data-run-feedback]');

  if (runButton) {
    runButton.addEventListener('click', async () => {
      if (runButton.disabled) {
        return;
      }

      runButton.disabled = true;
      spinner?.classList.remove('hidden');
      label && (label.textContent = 'Queued...');
      feedback && (feedback.textContent = 'Launching analytics pipeline. You can continue to browse while it runs.');

      try {
        const response = await fetch('/run-analyses', { method: 'POST' });
        const result = await response.json();

        if (result.status === 'already_running') {
          feedback && (feedback.textContent = 'Pipeline is already running. You will see new files once processing completes.');
        } else {
          label && (label.textContent = 'Running...');
          await pollUntilComplete();
          feedback && (feedback.textContent = 'Processing complete. Refreshing results.');
          await sleep(500);
          window.location.reload();
        }
      } catch (error) {
        console.error('Failed to run analyses', error);
        feedback && (feedback.textContent = 'Unable to launch the pipeline. Check server logs and try again.');
        runButton.disabled = false;
        spinner?.classList.add('hidden');
        label && (label.textContent = 'Run Full Pipeline');
      }
    });
  }
});

async function pollUntilComplete() {
  while (true) {
    try {
      const statusResponse = await fetch('/status');
      const status = await statusResponse.json();
      if (!status.running) {
        break;
      }
    } catch (error) {
      console.warn('Polling status failed', error);
    }
    await sleep(1000);
  }
}
