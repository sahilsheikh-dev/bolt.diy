import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';

export default class GoogleProvider extends BaseProvider {
  name = 'Google';
  getApiKeyLink = 'https://aistudio.google.com/app/apikey';

  config = {
    apiTokenKey: 'GOOGLE_GENERATIVE_AI_API_KEY',
  };

  staticModels: ModelInfo[] = [
    {
      name: 'gemini-2.5-pro',
      label: 'Gemini 2.5 Pro',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 65536,
    },
    {
      name: 'gemini-2.5-flash',
      label: 'Gemini 2.5 Flash',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 65536,
    },
  ];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv?: Record<string, string>,
  ): Promise<ModelInfo[]> {
    const { apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: '',
      defaultApiTokenKey: 'GOOGLE_GENERATIVE_AI_API_KEY',
    });

    if (!apiKey) {
      throw `Missing Api Key configuration for ${this.name} provider`;
    }

    // 👇 Use v1 or fallback to v1beta dynamically from env
    const API_VERSION = serverEnv?.GOOGLE_API_VERSION || 'v1beta';

    const BASE_URL = serverEnv?.GOOGLE_BASE_URL || 'https://generativelanguage.googleapis.com';
    const MODELS_ENDPOINT = `${BASE_URL}/${API_VERSION}/models?key=${apiKey}`;

    const response = await fetch(MODELS_ENDPOINT, {
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch models from Google API: ${response.status} ${response.statusText}`);
    }

    const body = await response.json();

    if (
      !body ||
      typeof body !== 'object' ||
      !('models' in body) ||
      !(body as any).models ||
      !Array.isArray((body as any).models)
    ) {
      throw new Error('Invalid response format from Google API');
    }

    const res = body as { models: any[] };

    const data = res.models.filter((model: any) => {
      const hasGoodTokenLimit = (model.outputTokenLimit || 0) > 8000;
      const isStable = !model.name.includes('exp') || model.name.includes('flash-exp');

      return hasGoodTokenLimit && isStable;
    });

    return data.map((m: any) => {
      const modelName = m.name.replace('models/', '');
      let contextWindow = m.inputTokenLimit || 32000;

      if (modelName.includes('gemini-1.5-pro')) {
        contextWindow = 2000000;
      }

      if (modelName.includes('gemini-1.5-flash')) {
        contextWindow = 1000000;
      }

      const completionTokens = Math.min(m.outputTokenLimit || 8192, 128000);

      return {
        name: modelName,
        label: `${m.displayName} (${contextWindow >= 1000000 ? Math.floor(contextWindow / 1000000) + 'M' : Math.floor(contextWindow / 1000) + 'k'} context)`,
        provider: this.name,
        maxTokenAllowed: contextWindow,
        maxCompletionTokens: completionTokens,
      };
    });
  }

  getModelInstance(options: {
    model: string;
    serverEnv: any;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: '',
      defaultApiTokenKey: 'GOOGLE_GENERATIVE_AI_API_KEY',
    });

    if (!apiKey) {
      throw new Error(`Missing API key for ${this.name} provider`);
    }

    const API_VERSION = serverEnv?.GOOGLE_API_VERSION || 'v1beta';

    const BASE_URL = serverEnv?.GOOGLE_BASE_URL || 'https://generativelanguage.googleapis.com';
    const ENDPOINT = `${BASE_URL}/${API_VERSION}`;

    const google = createGoogleGenerativeAI({ apiKey, baseUrl: ENDPOINT });
    const modelInstance: any = google(model);

    // Patch the internal fetch before any request happens
    const originalFetch = modelInstance.config.fetch ?? globalThis.fetch;

    modelInstance.config.fetch = async (url: string, opts: RequestInit) => {
      try {
        if (opts?.body && typeof opts.body === 'string' && opts.body.includes('"systemInstruction"')) {
          const parsed = JSON.parse(opts.body);

          if (parsed.systemInstruction) {
            // ✅ Gemini 2.x expects `system_instruction`, not `system`
            parsed.system_instruction = parsed.systemInstruction;
            delete parsed.systemInstruction;
            opts.body = JSON.stringify(parsed);
          }
        }
      } catch (e) {
        console.warn('Google fetch patch error:', e);
      }

      return originalFetch(url, opts);
    };

    return modelInstance;
  }
}
